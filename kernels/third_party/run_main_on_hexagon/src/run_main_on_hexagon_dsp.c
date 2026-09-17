/*==============================================================================
  Copyright (c) 2019-2020-2022 Qualcomm Technologies, Inc.
  All rights reserved. Qualcomm Proprietary and Confidential.
==============================================================================*/

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dlfcn.h>
#include "run_main_on_hexagon.h"
#include "HAP_farf.h"
#include "AEEStdErr.h"

#define BAIL_IF_ZERO(_err_, _x_) do { if (0 == (_x_)) { _err_ = 1; goto bail; } } while(0)
#define _FREE_(_x_) { if (_x_) { free(_x_); } }
#define MAX_ARGS 100

static void *DSO = 0;

// Deconstruct args string into discrete arguments and module name
static int parse_arguments(char* args, int* argc, char** argv)
{
   int nErr = AEE_SUCCESS, largc;
   const char delim[2] = " ";
   char* token, *strptr;

   BAIL_IF_ZERO(nErr, args);
   BAIL_IF_ZERO(nErr, argc);
   BAIL_IF_ZERO(nErr, argv);

   FARF(HIGH, "Arguments passed from AP: %s", args);
   token = strtok_r(args, delim, &strptr);
   BAIL_IF_ZERO(nErr, token);
   argv[0] = token;
   largc = 1;

   do {
      token = strtok_r(NULL, delim, &strptr);
      if (0 != token) {
          // Do not pass run_main_on_hexagon specific arguments to user main()
	  if(!(strncmp(token,"unsigned_pd",strlen("unsigned_pd"))) ||
		!(strncmp(token,"stack_size",strlen("stack_size"))) || !(strncmp(token,"domain_type",strlen("domain_type")))|| !(strncmp(token,"core_id",strlen("core_id"))) || !(strncmp(token,"loader_flag",strlen("loader_flag")))) {
		continue;
	  }
         argv[largc++] = token;
      }
   } while (0 != token);

   *argc = largc;

bail:
   return nErr;
}


struct thread_context {
   char* args;
   const char* module;
   int result;
#if (__clang_major__ > 12)
   unsigned int loader_flags;
#endif
};


static void *ribbon(void* pv)
{
   int (*func_ptr)(int argc, char* argv[]);
   int len = 0, argc = 0;
   char* args_local = 0;
   struct thread_context* context = (struct thread_context*)pv;
   char *args = context->args;
#if (__clang_major__ > 12)
   unsigned int loader_flags = context->loader_flags;
#endif
   context->result = 1;
   char* argv[MAX_ARGS];

   // Alloc local args string we can use to break into tokens
   len = strlen(args) + 1;
   args_local = (char*)malloc(len);
   if (0 == args_local) {
      FARF(ERROR, "failed to malloc %d bytes", len);
      return NULL;
   }
   if (strlcpy(args_local, args, len) >= len) {
      FARF(ERROR, "Buffer is truncated while copying args: %s", args);
      goto bail;
   }

   // Break apart arg string into discrete args
   if (0 != parse_arguments(args_local, &argc, argv)) {
      FARF(ERROR, "Failed to parse args: %s", args);
      goto bail;
   }

   FARF(HIGH, "args to main main()");
   for (int i = 0; i < argc; i++) {
      FARF(HIGH, "  argv[%d]: %s", i, argv[i]);
   }

#if (__clang_major__ > 12)
   // Open Module
   DSO = dlopen(context->module, loader_flags);
   if (0 == DSO) {
      FARF(ERROR, "failed to open module:%s", context->module);
      goto bail;
   }
#endif

   // Lookup main()
   func_ptr = (int (*)(int, char**))dlsym(DSO, "main");
   if (!func_ptr) {
      FARF(ERROR, "No main() found in module %s", context->module);
      goto bail;
   }

   // Call main()
   FARF(HIGH, "Calling main() in module %s", context->module);
   context->result = (*func_ptr)(argc, argv);
   FARF(HIGH, "main() returned %d", context->result);

bail:
#if (__clang_major__ > 12)
   // Close Module
   if (DSO)
      dlclose(DSO);
#endif

   _FREE_(args_local);
   return NULL;
}

int run_main_on_hexagon_doit(remote_handle64 handle64, unsigned int stack_size, unsigned int loader_flags, const char* args, int* result)
{
   int nErr = AEE_SUCCESS, len = 0;
   char* module;

   char* args_local = 0, *strptr;
   char* stack = 0;
   struct thread_context context;

   pthread_t pid = 0;
   pthread_attr_t thread_attr;
   int exit_status[1];
   int status;

   // Check args passed by user
   if (0 == args || 0 == strlen(args)) {
      FARF(ERROR, "Bad args");
      return AEE_EBADPARM;
   }

   // Alloc user passed stack for the thread
   stack = (char*)malloc(stack_size);
   if (0 == stack) {
      FARF(ERROR, "Failed to malloc stack size 0x%X", stack_size);
      return AEE_ENOMEMORY;
   }
   FARF(HIGH, " malloc stack size 0x%X", stack_size);

   // Alloc local args string we can use to break into tokens
   len = strlen(args) + 1;
   args_local = (char*)malloc(len);
   if (0 == args_local) {
      FARF(ERROR, "failed to malloc %d bytes", len);
      nErr = AEE_ENOMEMORY;
      goto bail;
   }
   if (strlcpy(args_local, args, len) >= len) {
      FARF(ERROR, "Buffer is truncated while copying args: %s", args);
      nErr = AEE_EBADCONTEXT;
      goto bail;
   }

   // Get Module
   module = strtok_r(args_local, " ", &strptr);
   if (0 == module) {
      FARF(ERROR, "module not found");
      nErr = AEE_EBADCONTEXT;
      goto bail;
   }
   context.args = (char *)args;
   context.module = module;

#if (__clang_major__ < 13)
   // Open Module
   DSO = dlopen(context.module, loader_flags);
   if (0 == DSO) {
      FARF(ERROR, "failed to open module:%s", context.module);
      nErr = AEE_EUNABLETOLOAD;
      goto bail;
   }
#else
   context.loader_flags = loader_flags;
#endif

   // Start ribbon thread
   status = pthread_attr_init(&thread_attr);
   if (status) {
      FARF(ERROR, "Error 0x%x from pthread_attr_init", status);
      nErr = AEE_EBADSTATE;
      goto bail;
   }

   pthread_attr_setthreadname(&thread_attr, "ribbon");
   pthread_attr_setstackaddr(&thread_attr,stack);
   pthread_attr_setstacksize(&thread_attr,stack_size);
   status = pthread_create(&pid, &thread_attr, ribbon, &context);

   if (status) {
      FARF(ERROR, "Error 0x%x from pthread_create", status);
      nErr = AEE_EBADSTATE;
      goto bail;
   }

   // Wait for ribbon to finish.
   status = pthread_join(pid, (void**) &exit_status);
   if (status) {
      FARF(ERROR, "Error 0x%x from pthread_join", status);
      nErr = AEE_EBADSTATE;
      goto bail;
   }
   FARF(HIGH, "Main() returned %d", context.result);
   *result = context.result;

bail:
#if (__clang_major__ < 13)
   // Close Module
   if (DSO)
      dlclose(DSO);
#endif
   // Clean up ribbon thread
   if (0 != (status = pthread_attr_destroy(&thread_attr))) {
      FARF(ERROR, "Error 0x%x from pthread_attr_destroy", status);
      nErr = AEE_EBADSTATE;
   }
   _FREE_(args_local);
   _FREE_(stack);
   return nErr;
}

/**
 * @param handle, the value returned by open
 * @retval, 0 for success, should always succeed
 */
int run_main_on_hexagon_close(remote_handle64 handle)
{
   _FREE_((void*)handle);
   return AEE_SUCCESS;
}

int run_main_on_hexagon_open(const char* uri, remote_handle64* handle)
{
   void* tptr = 0;
   /* Handle can take any value, RPC layer doesn't care
    * *handle = 0;
    * *handle = 0xdeadc0de;
    */
   tptr = (void*)malloc(1);
   if (0 == tptr) {
      FARF(ERROR, "failed to malloc 1 bytes");
      return AEE_ENOMEMORY;
   }
   *handle = (remote_handle64)tptr;

   return AEE_SUCCESS;
}
