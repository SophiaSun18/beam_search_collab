#pragma once

// =============================================================================
// qleaf_rabitq.cuh — RaBitQ edge-residual scorer.
//
// Header-only `__device__ __forceinline__` helper used by GATE=1 (local
// top-k / adaptive keep) and NAV=8 (Sophia edge-residual RaBitQ-as-key).
// The hot loop is a 6-plane × popcount evaluation against the per-query
// q6 bitplane upload and the per-edge 1-bit code.
//
// Algebraic shape (identical for both consumers):
//     est = approx_base + triple
//                       + factor_dq * width * signed_sum
//                       + factor_vq * lower
//     where signed_sum = 2 * Σ_{d : sign[d]=1} qquant[d]  - sum_q
//
// The helper is called from divergent lane masks (e.g. `if (lane_id==0)`
// blocks at frontier setup, and per-lane `if (valid)` blocks during
// candidate expansion). It MUST remain safe to call from any lane subset
// — no warp shuffles, no `__syncwarp`, no candidate buffer writes here.
// =============================================================================

#include "qleaf_config.cuh"

// =============================================================================
// Edge-residual RaBitQ bitplane scorer.
//
// Used by:
//   - NAV=8 (Sophia-style edge-residual RaBitQ-as-key) — approx_base is
//     parent exact L2 (computed by the caller).
//   - GATE_MODE=1 (local top-k / adaptive keep) — approx_base is
//     parent_nav or parent_dist depending on kRouteASortByNav.
//
// The helper computes only the score; it does not write candidates, decide
// keep, or touch hash. Caller passes the per-edge sidecar pointers and the
// per-query state (q6 bitplanes flattened to const uint32_t*; pass
// `&s_query_bp[0][0]`).
// =============================================================================

struct RouteAEdgeRabitqScore {
    float approx;
    bool ready;
};

static __device__ __forceinline__ RouteAEdgeRabitqScore
routea_edge_rabitq_score_bitplane(
    bool valid,
    bool routea_static_ready,
    size_t edge_pos,
    const uint32_t* __restrict__ d_sign_bit,
    int sign_word_stride,
    const float* __restrict__ d_rabitq_triple,
    const float* __restrict__ d_rabitq_factor_dq,
    const float* __restrict__ d_rabitq_factor_vq,
    const uint32_t* __restrict__ s_query_bp_flat,
    int rabitq_score_words,
    int rabitq_score_dims,
    int s_routea_query_sumq,
    float s_routea_query_width,
    float s_routea_query_lower,
    float approx_base) {
    RouteAEdgeRabitqScore out;
    out.approx = FLT_MAX;
    out.ready = false;

    if (!valid || !routea_static_ready ||
        d_sign_bit == nullptr ||
        d_rabitq_triple == nullptr ||
        d_rabitq_factor_dq == nullptr ||
        d_rabitq_factor_vq == nullptr ||
        sign_word_stride < rabitq_score_words ||
        rabitq_score_words <= 0) {
        return out;
    }

    const int tail_bits = rabitq_score_dims & 31;
    const uint32_t tail_mask =
        (tail_bits == 0) ? 0xFFFFFFFFu : ((1u << tail_bits) - 1u);
    const uint32_t* sign_words =
        d_sign_bit + edge_pos * static_cast<size_t>(sign_word_stride);

    int plane_sum0 = 0;
    int plane_sum1 = 0;
    int plane_sum2 = 0;
    int plane_sum3 = 0;
    int plane_sum4 = 0;
    int plane_sum5 = 0;

    #pragma unroll
    for (int w = 0; w < kStableRabitqScoreWords; ++w) {
        if (w < rabitq_score_words) {
            uint32_t sw = sign_words[w];
            if (tail_bits != 0 && w == rabitq_score_words - 1) {
                sw &= tail_mask;
            }
            plane_sum0 += __popc(s_query_bp_flat[0 * kStableRabitqScoreWords + w] & sw);
            plane_sum1 += __popc(s_query_bp_flat[1 * kStableRabitqScoreWords + w] & sw);
            plane_sum2 += __popc(s_query_bp_flat[2 * kStableRabitqScoreWords + w] & sw);
            plane_sum3 += __popc(s_query_bp_flat[3 * kStableRabitqScoreWords + w] & sw);
            plane_sum4 += __popc(s_query_bp_flat[4 * kStableRabitqScoreWords + w] & sw);
            plane_sum5 += __popc(s_query_bp_flat[5 * kStableRabitqScoreWords + w] & sw);
        }
    }

    const int set_sum =
        plane_sum0 + (plane_sum1 << 1) + (plane_sum2 << 2) +
        (plane_sum3 << 3) + (plane_sum4 << 4) + (plane_sum5 << 5);
    const int signed_sum = (set_sum << 1) - s_routea_query_sumq;
    const float triple = d_rabitq_triple[edge_pos];
    const float fdq = d_rabitq_factor_dq[edge_pos];
    const float fvq = d_rabitq_factor_vq[edge_pos];

    float approx = approx_base + triple +
                   (fdq * (s_routea_query_width * static_cast<float>(signed_sum))) +
                   (fvq * s_routea_query_lower);
    if (!(approx > 0.0f) || (approx != approx) || (approx >= FLT_MAX)) {
        approx = FLT_MAX * 0.5f;
    }

    out.approx = approx;
    out.ready = true;
    return out;
}
