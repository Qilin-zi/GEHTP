/*==============================================================================
Copyright (c) 2015 Qualcomm Technologies, Inc.
All rights reserved. Qualcomm Proprietary and Confidential.
==============================================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __LA_FLAG
#include <android/log.h>
#endif
#include <dlfcn.h>
#include "remote.h"
#include "run_main_on_hexagon.h"
#include "dsp_capabilities_utils.h"

#define MAX_ARGS_LEN 2048
#define DEFAULT_STACK_SIZE 1024*256

char args_buf[MAX_ARGS_LEN];

static void print_usage(char* program)
{
    printf("Usage:\n\n"
    "  %s domain path <stack_size=> <domain_type=> <core_id=> <args>\n\n"
    "  domain      : which Hexagon DSP to use, expressed as numeric domain id \n"
    "                supported domains: 0 (ADSP), 1 (MDSP), 2 (SDSP), 3 (CDSP)\n\n"
    "  path        : file path to .so that exports symbol that adheres to: \"int main(int argc, char *argv[])\"\n\n"
    "  stack_size= : optional argument to configure the stack size of run_main_on_hexagon thread that runs main()\n"
    "                default stack size is 256kb and stack size less than 256kb is not allowed \n\n"
    "  unsigned_pd= : optional argument to configure the type of PD (signed/unsigned) \n"
    "                 default value is 1 which makes unsigned offload \n"
    "                 pass unsigned_pd=0 if you want to invoke as signed PD\n\n"
    " domain_type=  : optional argument to run on domain of specified type.\n"
    "                 NSP: Run on NSP subsystem.\n"
    "                 HPASS: Run on HPASS subsystem.\n"
    " core_id=  : optional arguent to run on ith core starting from zero from the number of available cores of specified domain_type.\n"
    " loader_flag= : optional argument to configure loader flags.\n"
    "                       RTLD_LOCAL    - (Default) Symbols are hidden from other libraries.\n"
    "                       RTLD_GLOBAL   - Symbols are visible to subsequently loaded libraries.\n"
    "                       Note: RTLD_NOW is implicitly forced for thread safety.\n"

    "  args        : optional string arguments to pass to main()\n\n"
    "  e.g. %s 3 test_main.so stack_size=0x400000 1 foo2 2 bar\n\n"
    "     above command runs main() of test_main.so in a thread of stack size 0x400000 with arguments \"1 foo2 2 bar\" on CDSP\n\n"

    "  e.g. %s 0 test_main.so unsigned_pd=0 1 foo2 2 bar\n\n"
    "     above command runs main() of test_main.so with arguments \"1 foo2 2 bar\" in signed PD on ADSP\n\n"

    "  e.g. %s 3 test_main.so domain_type=NSP core_id=2 1 foo2 2 bar\n\n"
    "     above command runs main() of test_main.so with arguments \"1 foo2 2 bar\" in unsigned PD on 2nd core of type NSP\n" , program, program, program, program);
}

int main(int argc, char* argv[])
{
    int nErr = AEE_SUCCESS;
    char* stack_arg;
    char* unsigned_arg;
    unsigned int stack_size = 0, requested_pd = 1;
    int result = 0;
    int domain_id = -1;
    bool is_signedpd_requested = false;
    remote_handle64 handle64 = -1;
    int run_main_on_hexagon_URI_domain_len = strlen(run_main_on_hexagon_URI) + MAX_DOMAIN_NAMELEN;
    char* run_main_on_hexagon_URI_domain = NULL;
    domain* my_domain = NULL;
    int use_logical_id = 0;
    char domain_type_info[10];
    int core_id= -1;
    unsigned int loader_flags = RTLD_LOCAL;
    fastrpc_domain *domains_info = NULL;
    int num_domains = -1;
    char *uri = NULL;
    char* domain_type_arg = NULL;
    char* loader_flag_arg = NULL;
    char* core_id_arg = NULL;
    char* end;
    int length;
    int session_id = 0;
    char loader_flag_info[15];


    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }


    // concat args into string to send over the wire
    {
        int i = 0;
        int len = 0;
        char* ptr;
        size_t arg_size;

        args_buf[0] = '\0';
        for (i = 2 /* skip my name and domain */, ptr = args_buf; i < argc; i++) {
            arg_size = strlen(argv[i]) + 1;
            len += (arg_size);
            if (len >= MAX_ARGS_LEN) {
                print_usage(argv[0]);
                return -1;
            }
            snprintf(ptr, arg_size, "%s", argv[i]);

            ptr += (arg_size - 1);
            *ptr++ = ' ';
        }
        *ptr = '\0';
    }

    domain_id = atoi(argv[1]);

    domain_type_arg = strstr(args_buf, "domain_type=");
    if (domain_type_arg) {
        domain_type_arg += strlen("domain_type=");
        end = strchr(domain_type_arg, ' ');
        if (end != NULL) {
            length = end - domain_type_arg ;
            strncpy(domain_type_info, domain_type_arg, length);
            domain_type_info[length] = '\0'; // Null-terminate the result
            printf("Domain_type: %s\n", domain_type_info);
        }else {
            print_usage(argv[0]);
            return -1;
        }
        if((strcmp(domain_type_info, "NSP") != 0 && strcmp(domain_type_info, "HPASS") != 0 && strcmp(domain_type_info, "LPASS") != 0)) {
            printf("\nInvalid domain_type %s. Possible values are \"NSP\" or \"HPASS\" or \"LPASS\".",domain_type_info);
            nErr = AEE_EBADPARM;
            goto bail;
        }else {
            nErr = get_domains_info(domain_type_info,&num_domains,&domains_info);
            if (nErr == AEE_EUNSUPPORTED) {
                printf("Remote_system_request API is not supported on this target so cannot get domains info from the device. Falling back to legacy approach of using default domain id\n");
                nErr = get_dsp_support(&domain_id);
                if (nErr != AEE_SUCCESS) {
                    printf("ERROR in get_dsp_support: 0x%x, defaulting to CDSP domain\n", nErr);
                }
            }
            else if(nErr != AEE_SUCCESS) {
                printf("Error in getting domains information\n");
                goto bail;
            }
            else {
                core_id_arg = strstr(args_buf, "core_id=");
                if (core_id_arg) {
                    core_id_arg += strlen("core_id=");
                    core_id = (unsigned int)strtol(core_id_arg, NULL, 0);
                    printf("core_id: %d\n", core_id);
                    if (core_id < 0 || core_id >= num_domains) {
                        printf("Invalid core_id = %d for %s. Core_id should be between 0 to %d\n", core_id, domain_type_info, num_domains-1);
                        nErr = AEE_EBADPARM;
                        goto bail;
                    }
                }
                else {
                    core_id = 0;
                }
                use_logical_id = 1;
                nErr = get_effective_domain_id(domains_info[core_id].name, session_id, &domain_id);
                if (nErr != AEE_SUCCESS) {
                    printf("ERROR in get_effective_domain_id: 0x%x", nErr);
                    goto bail;
                }
            }
        }
    }

    loader_flag_arg = strstr(args_buf, "loader_flag=");

    if (loader_flag_arg) {
        loader_flag_arg += strlen("loader_flag=");
        if (*loader_flag_arg == '\0') {
            printf("\nError: loader_flag specified without a value.\n");
            nErr = AEE_EBADPARM;
            print_usage(argv[0]);
            goto bail;
        }
        end = strchr(loader_flag_arg, ' ');
        if (end != NULL)
            length = end - loader_flag_arg;
        else
            length = strlen(loader_flag_arg);
        if (length >= sizeof(loader_flag_info)) {
            printf("\nError: loader_flag value too long. Possible values are RTLD_GLOBAL or RTLD_LOCAL.\n");
            nErr = AEE_EBADPARM;
            goto bail;
        }
        strncpy(loader_flag_info, loader_flag_arg, length);
        loader_flag_info[length] = '\0';
        printf("loader_flag: %s\n", loader_flag_info);

        if (strcmp(loader_flag_info, "RTLD_GLOBAL") == 0)
            loader_flags = RTLD_GLOBAL;
        else if (strcmp(loader_flag_info, "RTLD_LOCAL") == 0)
            loader_flags = RTLD_LOCAL;
        else {
            printf("\nError: Invalid loader_flag %s. Possible values are RTLD_GLOBAL or RTLD_LOCAL.\n", loader_flag_info);
            nErr = AEE_EBADPARM;
            goto bail;
        }
    }

    loader_flags |= RTLD_NOW;

    if (use_logical_id == 0) {
        if (!is_valid_domain_id(domain_id, 0)) {
            nErr = AEE_EBADPARM;
            printf("\nERROR 0x%x: Invalid domain_id %d\n", nErr, domain_id);
            print_usage(argv[0]);
            goto bail;
        }
    }

    if (use_logical_id == 0) {
      my_domain = get_domain(domain_id);
      if (my_domain == NULL) {
        printf("\nERROR : unable to get domain struct %d\n",  domain_id);
        goto bail;
      }
      uri=my_domain->uri;
    }
    else{
        if ((uri = (char *)malloc(MAX_DOMAIN_NAMELEN)) == NULL) {
            nErr = AEE_ENOMEMORY;
            printf("unable to allocated memory for uri of size: %d", MAX_DOMAIN_NAMELEN);
            goto bail;
        }
        snprintf(uri, MAX_DOMAIN_NAMELEN, "%s%s", "&_dom=",domains_info[core_id].name);
    }

    if ((run_main_on_hexagon_URI_domain = (char *)malloc(run_main_on_hexagon_URI_domain_len)) == NULL) {
        nErr = AEE_ENOMEMORY;
        printf("unable to allocated memory for uri of size: %d", run_main_on_hexagon_URI_domain_len);
        goto bail;
    }

    nErr = snprintf(run_main_on_hexagon_URI_domain, run_main_on_hexagon_URI_domain_len, "%s%s", run_main_on_hexagon_URI, uri);
    if (nErr < 0) {
        printf("ERROR 0x%x returned from snprintf\n", nErr);
        nErr = AEE_EFAILED;
        goto bail;
    }

    // read optional stack size from user
    stack_arg = strstr(args_buf, "stack_size=");
    if (stack_arg) {
        stack_arg += strlen("stack_size=");
        stack_size = (unsigned int)strtol(stack_arg, NULL, 0);
    } else {
        stack_size = DEFAULT_STACK_SIZE;
    }
    if (stack_size < DEFAULT_STACK_SIZE){
        printf("Stack size less than 0x%X is not allowed\n",DEFAULT_STACK_SIZE);
        nErr = AEE_EBADPARM;
        goto bail;
    }

    //Read optional unsigned_pd argument
    unsigned_arg = strstr(args_buf, "unsigned_pd=");
    if (unsigned_arg) {
        unsigned_arg += strlen("unsigned_pd=");
        requested_pd = (unsigned int)strtol(unsigned_arg, NULL, 0);
    }

    if (requested_pd == 0){
        is_signedpd_requested = true;
    } else if (requested_pd == 1){
        is_signedpd_requested = false;
    } else {
        printf("invalid argument unsigned_pd=%s",unsigned_arg);
        print_usage(argv[3]);
        goto bail;
    }

    if(!is_signedpd_requested) {
        if (!is_unsignedpd_supported(domain_id)) {
            printf("Unsigned PD is not supported on domain %d.\n", domain_id);
            nErr = AEE_EFAILED;
            goto bail;
        }
    }

    printf("Attempting to run on %s PD on domain %d\n", is_signedpd_requested == true ? "signed":"unsigned", domain_id);


    if(&remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = domain_id;
        if(is_signedpd_requested){
            data.enable = 0;
        } else {
            data.enable = 1;
        }
        if (AEE_SUCCESS != (nErr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data)))) {
            printf("ERROR 0x%x: remote_session_control failed\n", nErr);
            goto bail;
        }
    }
    else {
        nErr = AEE_EUNSUPPORTED;
        printf("ERROR 0x%x: remote_session_control interface is not supported on this device\n", nErr);
        goto bail;
    }

    nErr = run_main_on_hexagon_open(run_main_on_hexagon_URI_domain, &handle64);
    if (nErr) {
        printf("Domain %d failed to open (0x%X), run_main_on_hexagon_URI_domain: %s\n", domain_id, nErr, run_main_on_hexagon_URI_domain);
        goto bail;
    }

    printf("RPC to Hexagon DSP with args: \"%s\"\n", args_buf);
    nErr = run_main_on_hexagon_doit(handle64, stack_size, loader_flags, args_buf, &result);

bail:
    if (run_main_on_hexagon_URI_domain) {
        free(run_main_on_hexagon_URI_domain);
    }
    if (uri && use_logical_id) {
        free(uri);
    }
    if (domains_info) {
        free(domains_info);
    }

    if (nErr) {
        printf("Error %d: Failed to call main() on DSP\n", nErr);
    } else if (result) {
        printf("Error: Main on Hexagon DSP returned %d\n", result);
    } else {
        printf("Successfully called main() on Hexagon DSP and received return value of 0.\n");
    }

    if (handle64) {
        if (run_main_on_hexagon_close(handle64)) {
            printf("Failed to close handle\n");
        }
    }
    return nErr;
}
