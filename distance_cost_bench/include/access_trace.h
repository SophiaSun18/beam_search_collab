#pragma once

#include <cstdint>

enum AccessTraceOp : uint32_t {
    ACCESS_TRACE_BASE_SEED_EXACT = 1,
    ACCESS_TRACE_BASE_EXPAND_EXACT = 2,
    ACCESS_TRACE_RQ_EXPAND = 3,
    ACCESS_TRACE_RQ_CHILD_EXACT = 4,
    ACCESS_TRACE_RQ_EXTRA_EXACT = 5,
    ACCESS_TRACE_RQ_SEED_EXACT = 6,
};

struct AccessTraceRecord {
    uint32_t query_id = 0;
    uint32_t step = 0;
    uint32_t op = 0;
    uint32_t node_id = 0;
    uint32_t active_mask = 0;
};

struct GpuFrontierAuditRecord {
    uint32_t query_id = 0;
    uint32_t step = 0;
    uint32_t chosen_node = 0;
    uint32_t chosen_exact_rank = 0;
    float frontier_exact_best_l2 = 0.0f;
    float frontier_chosen_exact_l2 = 0.0f;
    float frontier_exact_gap = 0.0f;
};

struct GpuQualityHitTraceRecord {
    uint32_t query_id = 0;
    uint32_t step = 0;
    uint8_t stage = 0;
    uint16_t new_gt_count = 0;
    uint16_t first_seen_gt_count = 0;
    uint16_t dropped_gt_count = 0;
    uint16_t current_hit_count = 0;
    uint16_t cumulative_seen_gt_count = 0;
    float topk_recall = 0.0f;
};
