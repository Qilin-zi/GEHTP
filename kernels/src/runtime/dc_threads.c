/* dc_threads.c — QURT 线程底座实现 (范式同 src/skel/main.c:660-682) */
#include "dc_threads.h"

#include <stdlib.h>
#include <HAP_farf.h>
#include <qurt_error.h>
#include <qurt.h>

int dc_spawn(dc_thread_t* t, const char* name, void (*fn)(void*), void* arg,
             uint32_t stack_bytes) {
    if (stack_bytes < 64u * 1024u) stack_bytes = 64u * 1024u;
    t->stack = memalign(4096, stack_bytes);
    if (!t->stack) return 0xD001;
    t->stack_bytes = stack_bytes;

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_name(&attr, name);
    qurt_thread_attr_set_priority(&attr, 10);       /* skel 同款 */
    qurt_thread_attr_set_stack_addr(&attr, t->stack);
    qurt_thread_attr_set_stack_size(&attr, stack_bytes);

    int rc = qurt_thread_create(&t->tid, &attr, fn, arg);
    if (rc != QURT_EOK) {
        FARF(ALWAYS, "dc_spawn(%s) FAIL rc=%d", name, rc);
        free(t->stack);
        t->stack = NULL;
        return 0xD002;
    }
    return 0;
}

void dc_join(dc_thread_t* t) {
    if (t->tid) {
        qurt_thread_join(t->tid, NULL);
        t->tid = 0;
    }
    if (t->stack) {
        free(t->stack);
        t->stack = NULL;
    }
}
