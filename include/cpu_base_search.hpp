#pragma once

#include "cpu_graph.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <unordered_set>
#include <vector>

struct CPUBaseSearchCfg {
    unsigned topk_size;
    vid_t    entry_point;
    int      max_iterations;
    bool     trace_first_query = false;
    int      trace_iterations = 0;
};

class CPUBaseSearch {
public:
    CPUBaseSearch(const CPUGraph& graph, const CPUBaseSearchCfg& cfg)
        : graph_(graph), cfg_(cfg) {}

    void search(const float* query, vid_t* results, int K, bool trace = false) const {
        struct BeamEntry {
            float dist;
            vid_t node;
            bool expanded;
        };

        auto distance = [&](vid_t node) {
            const auto vec = graph_.vect(node);
            float sum = 0.0f;
            for (int d = 0; d < graph_.dim; ++d) {
                const float diff = query[d] - vec[static_cast<size_t>(d)];
                sum += diff * diff;
            }
            return sum;
        };

        std::vector<BeamEntry> beam;
        beam.reserve(cfg_.topk_size + graph_.degree);
        std::unordered_set<vid_t> seen;
        seen.reserve(cfg_.topk_size * 4);

        beam.push_back({distance(cfg_.entry_point), cfg_.entry_point, false});
        seen.insert(cfg_.entry_point);

        for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
            std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
                if (lhs.dist != rhs.dist) return lhs.dist < rhs.dist;
                return lhs.node < rhs.node;
            });

            auto it = std::find_if(beam.begin(), beam.end(), [](const BeamEntry& entry) {
                return !entry.expanded;
            });
            if (it == beam.end()) break;

            const vid_t current = it->node;
            if (trace && iter < cfg_.trace_iterations) {
                std::cout << "cpu iter " << iter << " expand " << current << "\n";
            }
            it->expanded = true;
            for (vid_t child : graph_.nbrs(current)) {
                if (!seen.insert(child).second) continue;
                beam.push_back({distance(child), child, false});
            }

            if (beam.size() > cfg_.topk_size) {
                std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
                    if (lhs.dist != rhs.dist) return lhs.dist < rhs.dist;
                    return lhs.node < rhs.node;
                });
                beam.resize(cfg_.topk_size);
            }
        }

        std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
            if (lhs.dist != rhs.dist) return lhs.dist < rhs.dist;
            return lhs.node < rhs.node;
        });

        for (int i = 0; i < K; ++i) {
            results[i] = (i < static_cast<int>(beam.size())) ? beam[static_cast<size_t>(i)].node : UINT32_MAX;
        }
    }

private:
    const CPUGraph&     graph_;
    CPUBaseSearchCfg    cfg_;
};
