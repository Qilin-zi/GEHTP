/*==============================================================================
  Copyright (c) 2019, 2021 Qualcomm Technologies, Inc.
  All rights reserved. Qualcomm Proprietary and Confidential.
==============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "dlfcn.h"
#include "AEEStdErr.h"
#include "qurt.h"

#define FARF_ERROR 1
#define FARF_HIGH 1
#define FARF_MEDIUM 1
#include "HAP_farf.h"
#include "run_main_on_hexagon.h"

#define CRASH_REASON_LEN 100
#define CRASH_FUNCTION_LEN 100

static char crashed_function[CRASH_FUNCTION_LEN]="Unknown";
static char crash_Reason[CRASH_REASON_LEN]={0};
uint64 unknown = 0xDEADD00D;
Dl_info info;

extern void qurt_process_cmdline_get(char *buf, unsigned buf_siz);

#define MAX_BUF_SIZE 2048
#define DEFAULT_STACK_SIZE 1024*256

const char* DELIM = "--";
char args_buf[MAX_BUF_SIZE];
unsigned int stack_size = DEFAULT_STACK_SIZE;
unsigned int loader_flags = RTLD_LOCAL;

struct thread_context {
   char* args;
   int result;
};

// Find the string before "delim" from "str" string
char* find_string(char* str, const char* delim)
{
   char *p = NULL;
   p = strstr(str, delim);
   if (p != NULL) {
      *p = '\0';
   }
   return str;
}

// Find arguments to run_main executable from "str" string
int find_args(char* str)
{
   char* end;
   int length;
   int nErr = AEE_SUCCESS;
   char* stack_arg = NULL;
   char loader_flag_info[15];
   char* loader_flag_arg = NULL;

   // Read optional stack size argument from user
   stack_arg = strstr(str, "stack_size=");
   if (stack_arg != NULL) {
      stack_arg += strlen("stack_size=");
      stack_size = (int)strtol(stack_arg, NULL, 0);
   }

   if (stack_size < DEFAULT_STACK_SIZE) {
      FARF(HIGH, "Stack size less than 0x%x is not allowed", DEFAULT_STACK_SIZE);
      nErr = AEE_EBADPARM;
      goto bail;
   }

   loader_flag_arg = strstr(args_buf, "loader_flag=");

   if (loader_flag_arg) {
      loader_flag_arg += strlen("loader_flag=");
      if (*loader_flag_arg == '\0') {
         printf("\nError: loader_flag specified without a value.\n");
         nErr = AEE_EBADPARM;
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

bail:
   return nErr;
}

static void *run_main_exception_handler(void *arg)
{
   unsigned int error_thread = QURT_EFATAL;
   qurt_sysevent_error_1_t sys_err;
   unsigned int *lr = NULL;
   unsigned int *fp = NULL;
   unsigned int *pc = NULL;
   unsigned int diff = 0;
   int nRet = 0;
   const char *so_name = "Unknown";
   const char *so_base_addr = (const char*)(unknown);
   error_thread = qurt_exception_wait3(&sys_err,sizeof(sys_err));
   if (error_thread == QURT_EFATAL) {
      printf ("QURT error handler registration unsuccessfull\n");
      qurt_exception_raise_nonfatal(0);
   }
   // user application exited with a non-zero value
   if ((sys_err.cause & 0xFF) == 0x08) {
      printf("User Exit Code: 0x%x\n", sys_err.cause>>8);
      qurt_exception_raise_nonfatal(0xBEEFBEEFU);
      return NULL;
   }
   /* err.cause is a combination of cause and cause2 values, i.e
   cause = exception_tcb->error.cause2 << 8 | exception_tcb->error.cause
   Hence, masking the lower 8 bits of "err.cause" to get only the cause value*/
   switch(sys_err.cause & 0xFF){
      case 1:
         strlcpy(crash_Reason, "precise exception occurrence", CRASH_REASON_LEN);
         break;
      case 2:
         strlcpy(crash_Reason, "NMI occurrence", CRASH_REASON_LEN);
         break;
      case 3:
         strlcpy(crash_Reason, "TLBMISS RW occurrence", CRASH_REASON_LEN);
         break;
      case 4:
         strlcpy(crash_Reason, "Interrupt raised on reserved vector, should never happen", CRASH_REASON_LEN);
         break;
      case 5:
         strlcpy(crash_Reason, "Kernel Assert", CRASH_REASON_LEN);
         break;
      case 6:
         strlcpy(crash_Reason, "trap0(#num) called with unsupported num", CRASH_REASON_LEN);
         break;
      case 7:
         strlcpy(crash_Reason, "trap1 not supported. Using Trap1 causes this error", CRASH_REASON_LEN);
         break;
      case 10:
         strlcpy(crash_Reason, "TLBMISS X (execution)", CRASH_REASON_LEN);
         break;
      case 11:
         strlcpy(crash_Reason, "Running thread stopped due to Fatal error on other HW thread", CRASH_REASON_LEN);
         break;
      case 12:
         strlcpy(crash_Reason, "Application called qurt_fatal_exit()", CRASH_REASON_LEN);
         break;
      case 13:
         strlcpy(crash_Reason, "Kernel received an invalid L1 interrupt", CRASH_REASON_LEN);
         break;
      case 14:
         strlcpy(crash_Reason, "Kernel received an floating point error", CRASH_REASON_LEN);
         break;
      default:
         strlcpy(crash_Reason, "unknown reason", CRASH_REASON_LEN);
   }

   fp = (unsigned int*)sys_err.fp;
   lr = (unsigned int*)sys_err.lr;
   pc = (unsigned int*)sys_err.fault_pc;

   nRet = dladdr(pc, &info); // get the shared object crashed.
   if (nRet) { // dladdr returns 1 on success
      so_name = info.dli_fname;
      so_base_addr = info.dli_fbase;
      if(strncmp(info.dli_fname,"_rtld_anonymous",strlen("_rtld_anonymous")) == 0) {
         so_name = "run_main_on_hexagon";
      }
      snprintf(crashed_function, CRASH_FUNCTION_LEN, "[<%p>] %s+0x%X:     (%s)\n", pc, info.dli_sname, (unsigned int)pc - (unsigned int)info.dli_saddr, info.dli_fname);
   }
   printf ("#######################################################################\n");
   printf ("!!! Exception occurred\n");
   printf (" ------------------------------ Exception Details are furnished below ----------------------------------------------------\n");
   printf ("Crashed Reason \"%s\"\n",crash_Reason);
   printf ("Crashed Shared Object \"%s\" load address : 0x%p \n",so_name,so_base_addr);
   printf ("QuRT error code     0x%X\n", sys_err.cause);
   printf ("Thread ID           0x%X\n", error_thread);
   printf ("SP                  0x%x\n", sys_err.sp);
   printf ("FP                  0x%x\n", sys_err.fp);
   printf ("ELR                 0x%X\n", sys_err.fault_pc);
   printf ("BADVA               0x%x\n", sys_err.badva);
   printf ("LR                  0x%X\n", sys_err.lr);
   printf ("SSR                 0x%x\n", sys_err.ssr);

   printf("Call trace : \n");
   printf("%s",crashed_function);
   while(fp != NULL ){
      diff = (*fp) - (unsigned int)fp;
      if (!(diff >= 8 && diff <= 0xfff8 && (diff % 8 == 0)))
         break;
      nRet = dladdr(lr, &info); // get the call stack
      if (nRet) { // dladdr returns 1 on success
         if(strncmp(info.dli_fname, "_rtld_anonymous", strlen("_rtld_anonymous")) == 0) {
            info.dli_fname = "run_main_on_hexagon";
         }
         printf ("[<%p>] %s+0x%X:     (%s)\n", lr, info.dli_sname, (unsigned int)lr - (unsigned int)info.dli_saddr, info.dli_fname);
      }
      lr = fp + 1;
      lr = (unsigned int*)(*lr);
      fp = (unsigned int*)(*fp);
   }
   printf("----------------------------- End of Crash Report --------------------------------------------------\n");
   exit(1);
}
int main(void)
{
   int nErr = AEE_SUCCESS;
   struct thread_context context;
   char* args_to_doit = NULL, *args_to_sim = NULL;
   int status = -1; /* Return value from doit function running on DSP. */
   size_t len = 0;

   // Initialize the loader
   DL_vtbl vtbl = { sizeof(DL_vtbl), HAP_debug_v2 };
   char *builtin[] = { (char *)"libc.so", (char *)"libgcc.so" };
   if (0 == dlinitex(2, builtin, &vtbl)) {
      FARF(ERROR, "Failed to init loader");
      nErr = AEE_EFAILED;
      goto bail;
   }

   // Get the arguments
   args_buf[0] = '\0';
   qurt_process_cmdline_get(args_buf, MAX_BUF_SIZE);
   len = strlen(args_buf);
   if (len == 0) {
      FARF(ERROR, "Failed to get args from QuRT");
      nErr = AEE_EFAILED;
      goto bail;
   }
   FARF(HIGH, "Args from QuRT: %s", args_buf);

   // Save a copy of "Args from QuRT"
   args_to_doit = (char*)malloc(len+1);
   if (args_to_doit == NULL) {
      FARF(ERROR, "Malloc failed for user arguments");
      nErr = AEE_ENORPCMEMORY;
      goto bail;
   }

   if (strlcpy(args_to_doit, args_buf, len+1) > len) {
      FARF(ERROR, "Buffer is truncated while copying args_buf: %s", args_buf);
      nErr = AEE_EBUFFERTOOSMALL;
      goto bail;
   }

   // Find the string before "--" to find args to simulator executable
   args_to_sim = find_string(args_to_doit, DELIM);
   if (args_to_sim != NULL)
   {
      // Find the args to simulator executable
      nErr = find_args(args_to_sim);
      if (nErr != AEE_SUCCESS)
         goto bail;
   }

   // Find string after "--" to get the module and its arguments
   context.args = strstr(args_buf, DELIM);
   if (context.args != NULL) {
      context.args += strlen(DELIM);
   } else {
      context.args = "";
   }

   // Start run_main_exception_handler thread
   pthread_t pid = 0;
   pthread_attr_t thread_attr;
   status = pthread_attr_init(&thread_attr);
   if (status) {
      FARF(ERROR, "Error 0x%x from pthread_attr_init", status);
      nErr = AEE_EBADSTATE;
      goto bail;
   }
   pthread_attr_setthreadname(&thread_attr, "run_main_exception_handler");
   status = pthread_create(&pid, &thread_attr, run_main_exception_handler, NULL);
   if (status) {
      FARF(ERROR, "Error 0x%x from pthread_create", status);
      nErr = AEE_EBADSTATE;
      goto bail;
   }
   // Call the doit function on the DSP
   status = run_main_on_hexagon_doit(0, stack_size, loader_flags, context.args, &context.result);
   if ((int)status != AEE_SUCCESS) {
      FARF(ERROR, "Error 0x%x while attempting to call main() on dsp", status);
      nErr = status;
      goto bail;
   }
   FARF(HIGH, "Main() returned %d", context.result);

bail:
   if (args_to_doit)
      free(args_to_doit);
   if (nErr != AEE_SUCCESS) {
      FARF(ERROR, "run_main_on_hexagon returned 0x%x", nErr);
      return nErr;
   }
   return context.result;
}

