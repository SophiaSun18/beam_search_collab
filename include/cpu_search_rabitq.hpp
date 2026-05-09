#pragma once

#include "cpu_graph.hpp"
#include "rabitq.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

static constexpr int CPU_RABITQ_BQUERY = 6;

struct CPURabitqSearchCfg {
    unsigned topk_size;
    vid_t    entry_point;
    int      max_iterations;
};

class CPURabitqSearch {
public:
    CPURabitqSearch(const CPUGraph& graph, const CPURabitqData& rq,
                    const CPURabitqSearchCfg& cfg)
        : graph_(graph), rq_(rq), cfg_(cfg),
          padded_dim_(rq.padded_dim), words_per_plane_(rq.words_per_plane) {}

    void search(const float* query, vid_t* results, int K) const {
        int dim  = graph_.dim;
        int pdim = padded_dim_;
        int degree = graph_.degree;

        // ── rotate query: signs * WHT ────────────────────────────────
        std::vector<float> q_rot(pdim);
        for (int d = 0; d < dim; ++d)
            q_rot[d] = query[d] * rq_.signs[d];
        for (int d = dim; d < pdim; ++d)
            q_rot[d] = 0.0f;
        wht_inplace(q_rot.data(), pdim);

        // ── quantize ────────────────────────────────────��────────────
        float low = q_rot[0], high = q_rot[0];
        for (int d = 1; d < pdim; ++d) {
            low  = std::min(low,  q_rot[d]);
            high = std::max(high, q_rot[d]);
        }
        float w     = (high - low) / ((1 << CPU_RABITQ_BQUERY) - 1);
        float inv_w = 1.0f / w;

        std::vector<uint8_t> quant(pdim);
        int32_t sum_q = 0;
        for (int d = 0; d < pdim; ++d) {
            uint8_t q_val = static_cast<uint8_t>(
                lroundf((q_rot[d] - low) * inv_w + 0.5f));
            quant[d] = q_val;
            sum_q += q_val;
        }

        // ── exact L2 helper ──────────────────────────────────────────
        auto l2_exact = [&](vid_t node) {
            const auto vec = graph_.vect(node);
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = query[d] - vec[d];
                sum += diff * diff;
            }
            return sum;
        };

        // ── estimate distance for one neighbor ──────────────────────
        int wpp = words_per_plane_;
        auto estimate = [&](vid_t node, int nbr_idx, float exact_dist) {
            const float* factors = rq_.factors.data()
                + static_cast<size_t>(node) * 3 * degree;
            float tx  = factors[nbr_idx];
            float fdq = factors[degree + nbr_idx];
            float fvq = factors[2 * degree + nbr_idx];

            if (tx == FLT_MAX) return FLT_MAX;

            const uint32_t* nbr_codes = rq_.codes.data()
                + static_cast<size_t>(node) * degree * wpp
                + nbr_idx * wpp;

            uint32_t raw_sum = 0;
            for (int d = 0; d < pdim; ++d) {
                if ((nbr_codes[d / 32] >> (d % 32)) & 1)
                    raw_sum += quant[d];
            }

            float ip = static_cast<float>(
                (static_cast<int32_t>(raw_sum) << 1) - sum_q);
            float est = tx + exact_dist + fdq * w * ip + fvq * low;

            if (!std::isfinite(est)) return FLT_MAX;
            return est;
        };

        // ── beam search ──────────────────────────────────────────────
        struct BeamEntry {
            float dist;
            vid_t node;
            bool  expanded;
        };

        std::vector<BeamEntry> beam;
        beam.reserve(cfg_.topk_size + graph_.degree);
        std::unordered_set<vid_t> seen;
        seen.reserve(cfg_.topk_size * 4);

        beam.push_back({l2_exact(cfg_.entry_point), cfg_.entry_point, false});
        seen.insert(cfg_.entry_point);

        auto beam_sort = [](const BeamEntry& a, const BeamEntry& b) {
            return (a.dist != b.dist) ? a.dist < b.dist : a.node < b.node;
        };

        for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
            std::sort(beam.begin(), beam.end(), beam_sort);

            auto it = std::find_if(beam.begin(), beam.end(),
                                   [](const BeamEntry& e) { return !e.expanded; });
            if (it == beam.end()) break;

            vid_t current = it->node;
            it->expanded = true;

            float exact_dist = l2_exact(current);
            auto nbrs = graph_.nbrs(current);
            for (int i = 0; i < degree; ++i) {
                vid_t child = nbrs[i];
                if (child == UINT32_MAX) continue;
                if (!seen.insert(child).second) continue;
                float est = estimate(current, i, exact_dist);
                beam.push_back({est, child, false});
            }

            if (beam.size() > cfg_.topk_size) {
                std::sort(beam.begin(), beam.end(), beam_sort);
                beam.resize(cfg_.topk_size);
            }
        }

        for (auto& e : beam)
            e.dist = l2_exact(e.node);
        std::sort(beam.begin(), beam.end(), beam_sort);
        for (int i = 0; i < K; ++i)
            results[i] = (i < static_cast<int>(beam.size()))
                ? beam[i].node : UINT32_MAX;
    }

private:
    static void wht_inplace(float* data, int n) {
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = 0; j < len; ++j) {
                    float u = data[i + j], v = data[i + j + len];
                    data[i + j]       = u + v;
                    data[i + j + len] = u - v;
                }
    }

    const CPUGraph&       graph_;
    const CPURabitqData&  rq_;
    CPURabitqSearchCfg    cfg_;
    int                   padded_dim_;
    int                   words_per_plane_;
};
