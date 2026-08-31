/* wpool.c — U12 常驻工人池 (见 include/wpool.h) */
#include "wpool.h"
#include <string.h>

static void wpool_worker(void* arg) {
    struct wpool* p = (struct wpool*)arg;
    for (;;) {
        wpool_job_fn fn = 0;
        void* ud = 0;
        dc_mutex_lock(&p->mx);
        if (p->qcount > 0) {
            fn = p->qfn[p->qh];
            ud = p->qud[p->qh];
            p->qh = (p->qh + 1) % WPOOL_QUEUE_DEPTH;
            p->qcount--;
            p->executed++;
        }
        dc_mutex_unlock(&p->mx);
        if (fn) {
            fn(ud);
            dc_mutex_lock(&p->mx);
            p->done++;
            dc_mutex_unlock(&p->mx);
        } else {
            dc_mutex_lock(&p->mx);
            uint32_t stop = p->stop;
            dc_mutex_unlock(&p->mx);
            if (stop) break;
        }
    }
}

int wpool_open(struct wpool* p, int nworkers, uint32_t stack_bytes) {
    if (!p || nworkers < 1 || nworkers > WPOOL_MAX_WORKERS) return -1;
    memset(p, 0, sizeof(*p));
    if (stack_bytes < 64 * 1024) stack_bytes = 64 * 1024;
    p->nworkers = nworkers;
    dc_mutex_init(&p->mx);
    for (int i = 0; i < nworkers; i++) {
        if (dc_spawn(&p->th[i], "wpool_w", wpool_worker, p, stack_bytes)) return -2;
        p->spawned = i + 1;
    }
    return 0;
}

int wpool_submit(struct wpool* p, wpool_job_fn fn, void* ud) {
    int rc = -1;
    dc_mutex_lock(&p->mx);
    if (p->qcount < WPOOL_QUEUE_DEPTH) {
        p->qfn[p->qt] = fn;
        p->qud[p->qt] = ud;
        p->qt = (p->qt + 1) % WPOOL_QUEUE_DEPTH;
        p->qcount++;
        rc = 0;
    }
    dc_mutex_unlock(&p->mx);
    return rc;
}

void wpool_wait_all(struct wpool* p, uint32_t njobs) {
    for (;;) {
        dc_mutex_lock(&p->mx);
        uint32_t d = p->done;
        dc_mutex_unlock(&p->mx);
        if (d >= njobs) break;
    }
}

void wpool_close(struct wpool* p) {
    if (!p || !p->spawned) return;
    for (;;) {
        dc_mutex_lock(&p->mx);
        uint32_t empty = (p->qcount == 0);
        dc_mutex_unlock(&p->mx);
        if (empty) break;
    }
    dc_mutex_lock(&p->mx);
    p->stop = 1;
    dc_mutex_unlock(&p->mx);
    for (int i = 0; i < p->spawned; i++) dc_join(&p->th[i]);
    p->spawned = 0;
}
