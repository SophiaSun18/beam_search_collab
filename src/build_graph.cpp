#include <algorithm>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

using vid_t = uint32_t;

namespace {

struct BuildConfig {
    std::string base_path;
    std::string graph_path;
    int degree = 32;
    int beam_width = 128;
    double alpha = 2.0;
    bool use_inner_product = false;
    uint32_t seed = 1234;
};

struct Dataset {
    size_t num_points = 0;
    int dim = 0;
    std::vector<float> values;
};

struct BeamItem {
    float dist = 0.0f;
    vid_t node = 0;
    bool expanded = false;
};

class BeamQueue {
public:
    explicit BeamQueue(size_t capacity) : capacity_(capacity) {}

    void push(vid_t node, float dist) {
        for (BeamItem& item : items_) {
            if (item.node == node) {
                if (dist < item.dist) item.dist = dist;
                return;
            }
        }
        items_.push_back({dist, node, false});
        std::sort(items_.begin(), items_.end(), [](const BeamItem& lhs, const BeamItem& rhs) {
            if (lhs.dist != rhs.dist) return lhs.dist < rhs.dist;
            return lhs.node < rhs.node;
        });
        if (items_.size() > capacity_) items_.resize(capacity_);
    }

    [[nodiscard]] bool has_unexpanded() const {
        for (const BeamItem& item : items_) {
            if (!item.expanded) return true;
        }
        return false;
    }

    [[nodiscard]] size_t next_unexpanded_index() const {
        for (size_t i = 0; i < items_.size(); ++i) {
            if (!items_[i].expanded) return i;
        }
        return items_.size();
    }

    void mark_expanded(size_t index) {
        items_[index].expanded = true;
    }

    [[nodiscard]] size_t size() const { return items_.size(); }
    [[nodiscard]] float tail_dist() const { return items_.back().dist; }
    [[nodiscard]] const BeamItem& operator[](size_t index) const { return items_[index]; }

private:
    size_t capacity_;
    std::vector<BeamItem> items_;
};

[[nodiscard]] const float* row_ptr(const Dataset& ds, size_t index) {
    return ds.values.data() + index * static_cast<size_t>(ds.dim);
}

[[nodiscard]] float l2_distance_squared(const float* lhs, const float* rhs, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    return sum;
}

[[nodiscard]] float inner_product_distance(const float* lhs, const float* rhs, int dim) {
    float dot = 0.0f;
    for (int i = 0; i < dim; ++i) dot += lhs[i] * rhs[i];
    return -dot;
}

[[nodiscard]] Dataset read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open base vectors: " + path);

    int dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim <= 0) throw std::runtime_error("Invalid fvecs header in: " + path);

    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t stride = sizeof(int) + static_cast<size_t>(dim) * sizeof(float);
    if (file_size % stride != 0) throw std::runtime_error("Corrupt fvecs file: " + path);
    const size_t num_points = file_size / stride;

    in.seekg(0, std::ios::beg);
    Dataset ds;
    ds.num_points = num_points;
    ds.dim = dim;
    ds.values.resize(num_points * static_cast<size_t>(dim));

    for (size_t i = 0; i < num_points; ++i) {
        int cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), sizeof(cur_dim));
        if (!in || cur_dim != dim) throw std::runtime_error("Inconsistent fvecs row in: " + path);
        in.read(reinterpret_cast<char*>(ds.values.data() + i * static_cast<size_t>(dim)),
                static_cast<std::streamsize>(static_cast<size_t>(dim) * sizeof(float)));
        if (!in) throw std::runtime_error("Failed reading vector payload from: " + path);
    }

    return ds;
}

void write_graph_bin(const std::string& path, size_t num_points, int degree,
                     const std::vector<vid_t>& edges) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output graph file: " + path);

    const int32_t n32 = static_cast<int32_t>(num_points);
    const int32_t d32 = degree;
    out.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
    out.write(reinterpret_cast<const char*>(&d32), sizeof(d32));
    out.write(reinterpret_cast<const char*>(edges.data()),
              static_cast<std::streamsize>(edges.size() * sizeof(vid_t)));
    if (!out) throw std::runtime_error("Failed writing graph file: " + path);
}

void dedupe_and_prune_neighbors(std::vector<std::pair<float, vid_t>>& neighbors,
                                int src,
                                int degree,
                                double alpha,
                                const Dataset& dataset,
                                bool use_inner_product) {
    if (neighbors.empty()) return;

    std::sort(neighbors.begin(), neighbors.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) return lhs.first < rhs.first;
        return lhs.second < rhs.second;
    });

    std::vector<std::pair<float, vid_t>> unique;
    unique.reserve(neighbors.size());
    vid_t last = std::numeric_limits<vid_t>::max();
    for (const auto& entry : neighbors) {
        if (entry.second == static_cast<vid_t>(src)) continue;
        if (entry.second == last) continue;
        unique.push_back(entry);
        last = entry.second;
    }

    if (unique.size() <= static_cast<size_t>(degree)) {
        neighbors.swap(unique);
        return;
    }

    auto euclidean_distance = [&](vid_t lhs, vid_t rhs) {
        return l2_distance_squared(row_ptr(dataset, lhs), row_ptr(dataset, rhs), dataset.dim);
    };

    std::vector<std::pair<float, vid_t>> pruned;
    pruned.reserve(degree);
    for (double current_alpha : {1.0, alpha}) {
        for (const auto& [dist0, node] : unique) {
            if (pruned.size() == static_cast<size_t>(degree)) break;
            float dist = dist0;
            if (use_inner_product) dist = euclidean_distance(static_cast<vid_t>(src), node);
            bool dominated = false;
            for (const auto& [kept_dist, kept] : pruned) {
                (void)kept_dist;
                if (static_cast<float>(current_alpha) * euclidean_distance(kept, node) <= dist) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) pruned.emplace_back(dist, node);
        }
    }

    neighbors.swap(pruned);
}

[[nodiscard]] BuildConfig parse_args(int argc, char** argv) {
    BuildConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for argument: " + std::string(name));
            return argv[++i];
        };

        if (arg == "--base") {
            cfg.base_path = require_value(arg);
        } else if (arg == "--graph") {
            cfg.graph_path = require_value(arg);
        } else if (arg == "--degree") {
            cfg.degree = std::stoi(require_value(arg));
        } else if (arg == "--beam-width") {
            cfg.beam_width = std::stoi(require_value(arg));
        } else if (arg == "--alpha") {
            cfg.alpha = std::stod(require_value(arg));
        } else if (arg == "--metric") {
            const std::string metric = require_value(arg);
            if (metric == "l2") cfg.use_inner_product = false;
            else if (metric == "ip") cfg.use_inner_product = true;
            else throw std::runtime_error("Unsupported metric: " + metric);
        } else if (arg == "--seed") {
            cfg.seed = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--help") {
            std::cout
                << "Usage: build_graph --base base.fvecs --graph graph.bin [options]\n"
                << "  --degree <int>       fixed out-degree (default: 32)\n"
                << "  --beam-width <int>   build beam width L (default: 128)\n"
                << "  --alpha <float>      Vamana pruning alpha (default: 2.0)\n"
                << "  --metric <l2|ip>     build distance metric (default: l2)\n"
                << "  --seed <int>         shuffle seed (default: 1234)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + std::string(arg));
        }
    }

    if (cfg.base_path.empty()) throw std::runtime_error("Missing required argument: --base");
    if (cfg.graph_path.empty()) throw std::runtime_error("Missing required argument: --graph");
    if (cfg.degree <= 0) throw std::runtime_error("--degree must be positive");
    if (cfg.beam_width <= 0) throw std::runtime_error("--beam-width must be positive");
    if (cfg.alpha < 1.0) throw std::runtime_error("--alpha must be at least 1.0");
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto build_start = std::chrono::steady_clock::now();
        const BuildConfig cfg = parse_args(argc, argv);
        const Dataset dataset = read_fvecs(cfg.base_path);

        const size_t num_points = dataset.num_points;
        const int dim = dataset.dim;
        const int degree = cfg.degree;
        const int beam_width = cfg.beam_width;
        const bool use_inner_product = cfg.use_inner_product;

        if (num_points == 0) throw std::runtime_error("Base dataset is empty");

        std::cout << "Building Vamana graph: n=" << num_points
                  << " dim=" << dim
                  << " degree=" << degree
                  << " beam_width=" << beam_width
                  << " alpha=" << cfg.alpha
                  << " metric=" << (use_inner_product ? "ip" : "l2")
                  << "\n";

        auto distance = [&](vid_t lhs, vid_t rhs) {
            const float* lhs_ptr = row_ptr(dataset, lhs);
            const float* rhs_ptr = row_ptr(dataset, rhs);
            if (use_inner_product) return inner_product_distance(lhs_ptr, rhs_ptr, dim);
            return l2_distance_squared(lhs_ptr, rhs_ptr, dim);
        };

        const size_t batch_bound = std::max<size_t>(1, static_cast<size_t>(0.02 * static_cast<double>(num_points)));
        size_t start = 1;
        std::vector<std::vector<std::pair<float, vid_t>>> graph(num_points);
        std::vector<vid_t> order(num_points);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 gen(cfg.seed);
        std::shuffle(order.begin(), order.end(), gen);

        while (start < num_points) {
            const size_t end = std::min(std::min(start * 2, start + batch_bound), num_points);
            const size_t batch_size = end - start;
            const size_t progress_step = std::max<size_t>(1, batch_size / 10);
            const auto batch_start = std::chrono::steady_clock::now();
            std::atomic<size_t> completed_in_batch{0};
            std::cout << "  start batch [" << start << ", " << end << ")"
                      << " size=" << batch_size << "\n";
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = start; i < end; ++i) {
                const vid_t target = order[i];
                const size_t hash_size = std::min(static_cast<size_t>(beam_width) * static_cast<size_t>(beam_width), num_points);
                std::unordered_set<vid_t> visited;
                visited.reserve(hash_size == 0 ? 1 : hash_size);

                BeamQueue beam(static_cast<size_t>(beam_width));
                beam.push(order[0], distance(order[0], target));
                visited.insert(order[0]);

                std::vector<vid_t> frontier;
                while (beam.has_unexpanded()) {
                    const size_t next = beam.next_unexpanded_index();
                    const BeamItem item = beam[next];
                    graph[target].emplace_back(item.dist, item.node);
                    beam.mark_expanded(next);

                    frontier.clear();
                    for (const auto& neighbor : graph[item.node]) {
                        if (visited.insert(neighbor.second).second) frontier.push_back(neighbor.second);
                    }
                    for (vid_t node : frontier) {
                        const float dist = distance(node, target);
                        if (beam.size() < static_cast<size_t>(beam_width) || dist < beam.tail_dist()) {
                            beam.push(node, dist);
                        }
                    }
                }

                dedupe_and_prune_neighbors(graph[target], static_cast<int>(target), degree,
                                           cfg.alpha, dataset, use_inner_product);

                const size_t built_in_batch = completed_in_batch.fetch_add(1, std::memory_order_relaxed) + 1;
                if (built_in_batch == batch_size || built_in_batch % progress_step == 0) {
                    const auto now = std::chrono::steady_clock::now();
                    const double batch_seconds = std::chrono::duration<double>(now - batch_start).count();
                    const double total_seconds = std::chrono::duration<double>(now - build_start).count();
                    const double pct_batch = 100.0 * static_cast<double>(built_in_batch) / static_cast<double>(batch_size);
                    const double pct_total = 100.0 * static_cast<double>(start + built_in_batch) / static_cast<double>(num_points);
                    #pragma omp critical(build_graph_progress)
                    std::cout << "    progress: batch=" << built_in_batch << "/" << batch_size
                              << " (" << pct_batch << "%)"
                              << " total=" << (start + built_in_batch) << "/" << num_points
                              << " (" << pct_total << "%)"
                              << " elapsed=" << total_seconds << "s"
                              << " batch_elapsed=" << batch_seconds << "s\n";
                }
            }

            for (size_t i = start; i < end; ++i) {
                const vid_t src = order[i];
                for (const auto& edge : graph[src]) {
                    graph[edge.second].emplace_back(edge.first, src);
                }
            }
            #pragma omp parallel for schedule(static)
            for (size_t u = 0; u < num_points; ++u) {
                if (graph[u].size() > static_cast<size_t>(degree)) {
                    dedupe_and_prune_neighbors(graph[u], static_cast<int>(u), degree,
                                               cfg.alpha, dataset, use_inner_product);
                }
            }

            const double batch_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - batch_start).count();
            std::cout << "  finished batch [" << start << ", " << end << ")"
                      << " in " << batch_seconds << "s\n";
            start = end;
        }

        std::vector<vid_t> edges(num_points * static_cast<size_t>(degree), 0);
        #pragma omp parallel for schedule(static)
        for (size_t node = 0; node < num_points; ++node) {
            dedupe_and_prune_neighbors(graph[node], static_cast<int>(node), degree,
                                       cfg.alpha, dataset, use_inner_product);
            const size_t limit = std::min(graph[node].size(), static_cast<size_t>(degree));
            for (size_t j = 0; j < limit; ++j) {
                edges[node * static_cast<size_t>(degree) + j] = graph[node][j].second;
            }
        }

        write_graph_bin(cfg.graph_path, num_points, degree, edges);
        const double total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - build_start).count();
        std::cout << "Wrote graph to " << cfg.graph_path << "\n";
        std::cout << "Total build time: " << total_seconds << "s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "build_graph error: " << e.what() << "\n";
        return 1;
    }
}
