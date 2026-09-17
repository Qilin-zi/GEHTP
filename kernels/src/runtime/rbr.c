/* rbr.c — U18 recurrent 状态部分接受回退 (见 include/rbr.h) */
#include "rbr.h"
#include <stdlib.h>
#include <string.h>

int rbr_init(struct rbr* r) {
    if (!r) return RBR_ERR_PARAM;
    memset(r, 0, sizeof(*r));
    return RBR_OK;
}

int rbr_register(struct rbr* r, uint32_t group, void* in_state, uint32_t bytes) {
    if (!r || !in_state || bytes == 0) return RBR_ERR_PARAM;
    if (r->has_snapshot) return RBR_ERR_FROZEN;      /* 布局稳定假设, §13 */
    if (group >= RBR_MAX_GROUPS) return RBR_ERR_FULL;
    if (r->grp[group].n_states >= RBR_MAX_STATES) return RBR_ERR_FULL;
    uint32_t idx = r->grp[group].n_states;
    r->grp[group].in[idx] = in_state;
    r->grp[group].bytes[idx] = bytes;
    r->grp[group].n_states++;
    if (group + 1 > r->n_groups) r->n_groups = group + 1;
    return RBR_OK;
}

int rbr_snapshot(struct rbr* r) {
    if (!r || r->n_groups == 0) return RBR_ERR_PARAM;
    for (uint32_t g = 0; g < r->n_groups; g++)
        for (uint32_t s = 0; s < r->grp[g].n_states; s++) {
            if (!r->grp[g].shadow[s]) {
                r->grp[g].shadow[s] = (uint8_t*)malloc(r->grp[g].bytes[s]);
                if (!r->grp[g].shadow[s]) return RBR_ERR_PARAM;
            }
            memcpy(r->grp[g].shadow[s], r->grp[g].in[s], r->grp[g].bytes[s]);
        }
    r->has_snapshot = 1;
    r->snap_frame = r->frame;
    r->n_snapshot++;
    return RBR_OK;
}

int rbr_restore(struct rbr* r) {
    if (!r) return RBR_ERR_PARAM;
    if (!r->has_snapshot) return RBR_ERR_NOSNAP;
    if (r->frame != r->snap_frame) return RBR_ERR_STALE;
    for (uint32_t g = 0; g < r->n_groups; g++)
        for (uint32_t s = 0; s < r->grp[g].n_states; s++)
            memcpy(r->grp[g].in[s], r->grp[g].shadow[s], r->grp[g].bytes[s]);
    r->skip_next = 1;                          /* INV-3 */
    r->n_restore++;
    return RBR_OK;
}

enum rbr_setup_mode rbr_setup_hook(struct rbr* r, uint32_t engine_n_past) {
    if (!r) return RBR_CLEAR;
    if (engine_n_past == 0) { r->skip_next = 0; return RBR_CLEAR; }
    if (r->skip_next) {
        r->skip_next = 0;
        r->n_skip_used++;
        return RBR_SKIP;
    }
    return RBR_COPY;
}

int rbr_note_process(struct rbr* r, uint32_t n_tokens) {
    if (!r || n_tokens == 0) return RBR_ERR_PARAM;
    r->frame++;
    r->n_past += n_tokens;
    r->last_tokens = n_tokens;
    return RBR_OK;
}

int rbr_rewind(struct rbr* r, uint32_t n_selected) {
    if (!r) return RBR_ERR_PARAM;
    if (!r->has_snapshot) return RBR_ERR_NOSNAP;
    if (n_selected == 0 || n_selected > r->last_tokens) return RBR_ERR_PARAM;
    r->n_past -= r->last_tokens;               /* 回本轮起点; replay 覆写 */
    r->n_rewind++;
    return RBR_OK;
}

uint32_t rbr_n_past(const struct rbr* r) { return r ? r->n_past : 0u; }

int rbr_shadow_equals(const struct rbr* r) {
    if (!r || !r->has_snapshot) return 0;
    for (uint32_t g = 0; g < r->n_groups; g++)
        for (uint32_t s = 0; s < r->grp[g].n_states; s++)
            if (memcmp(r->grp[g].in[s], r->grp[g].shadow[s],
                       r->grp[g].bytes[s]) != 0) return 0;
    return 1;
}

void rbr_close(struct rbr* r) {
    if (!r) return;
    for (uint32_t g = 0; g < RBR_MAX_GROUPS; g++)
        for (uint32_t s = 0; s < RBR_MAX_STATES; s++)
            if (r->grp[g].shadow[s]) {
                free(r->grp[g].shadow[s]);
                r->grp[g].shadow[s] = 0;
            }
    memset(r, 0, sizeof(*r));
}
