#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/utils.h"
#include "../include/qg.hpp"

using namespace std;

struct ReplayStats {
    uint64_t records = 0;
    uint64_t base_exact_groups = 0;
    uint64_t base_exact_ops = 0;
    double base_exact_ns = 0.0;

    uint64_t rq_expand_groups = 0;
    uint64_t rq_expand_exact_ops = 0;
    uint64_t rq_expand_est_ops = 0;
    double rq_expand_ns = 0.0;

    uint64_t rq_seed_exact_groups = 0;
    uint64_t rq_seed_exact_ops = 0;
    double rq_seed_exact_ns = 0.0;

    uint64_t rq_child_exact_groups = 0;
    uint64_t rq_child_exact_ops = 0;
    double rq_child_exact_ns = 0.0;

    uint64_t rq_extra_exact_groups = 0;
    uint64_t rq_extra_exact_ops = 0;
    double rq_extra_exact_ns = 0.0;

    uint64_t rq_query_prepares = 0;
    double rq_query_prepare_ns = 0.0;
};

static inline uint32_t trace_popcount(uint32_t mask, int degree) {
    if (degree >= 32) {
        return static_cast<uint32_t>(__builtin_popcount(mask));
    }
    const uint32_t valid = degree == 32 ? 0xffffffffu : ((1u << degree) - 1u);
    return static_cast<uint32_t>(__builtin_popcount(mask & valid));
}

static inline bool trace_has_op(const vector<AccessTraceRecord>& records, uint32_t op) {
    for (const auto& r : records) {
        if (r.op == op) return true;
    }
    return false;
}

static inline bool trace_has_rabitq_op(const vector<AccessTraceRecord>& records) {
    return trace_has_op(records, ACCESS_TRACE_RQ_EXPAND) ||
           trace_has_op(records, ACCESS_TRACE_RQ_CHILD_EXACT) ||
           trace_has_op(records, ACCESS_TRACE_RQ_EXTRA_EXACT) ||
           trace_has_op(records, ACCESS_TRACE_RQ_SEED_EXACT);
}

static inline bool trace_has_base_op(const vector<AccessTraceRecord>& records) {
    return trace_has_op(records, ACCESS_TRACE_BASE_SEED_EXACT) ||
           trace_has_op(records, ACCESS_TRACE_BASE_EXPAND_EXACT);
}

static void print_row(const char* device, const char* category,
                      uint64_t groups, uint64_t exact_ops, uint64_t est_ops,
                      double total_ns) {
    const uint64_t total_ops = exact_ops + est_ops;
    const double ns_per_group = groups ? total_ns / static_cast<double>(groups) : 0.0;
    const double ns_per_op = total_ops ? total_ns / static_cast<double>(total_ops) : 0.0;
    printf("%s,%s,groups=%llu,exact_ops=%llu,est_ops=%llu,total_ns=%.4f,ns_per_group=%.4f,ns_per_op=%.4f\n",
           device, category,
           static_cast<unsigned long long>(groups),
           static_cast<unsigned long long>(exact_ops),
           static_cast<unsigned long long>(est_ops),
           total_ns, ns_per_group, ns_per_op);
}

static void write_csv(const string& path, const ReplayStats& s) {
    ofstream out(path);
    if (!out.is_open()) {
        throw runtime_error("cannot open replay csv for writing: " + path);
    }
    out << "device,category,groups,exact_ops,est_ops,total_ns,ns_per_group,ns_per_op\n";

    auto emit = [&](const char* category, uint64_t groups,
                    uint64_t exact_ops, uint64_t est_ops, double total_ns) {
        const uint64_t total_ops = exact_ops + est_ops;
        out << "cpu," << category << ","
            << groups << ","
            << exact_ops << ","
            << est_ops << ","
            << total_ns << ","
            << (groups ? total_ns / static_cast<double>(groups) : 0.0) << ","
            << (total_ops ? total_ns / static_cast<double>(total_ops) : 0.0) << "\n";
    };

    emit("base_expand_exact", s.base_exact_groups, s.base_exact_ops, 0, s.base_exact_ns);
    emit("rq_expand_bundle", s.rq_expand_groups, s.rq_expand_exact_ops,
         s.rq_expand_est_ops, s.rq_expand_ns);
    emit("rq_seed_exact", s.rq_seed_exact_groups, s.rq_seed_exact_ops, 0,
         s.rq_seed_exact_ns);
    emit("rq_child_exact", s.rq_child_exact_groups, s.rq_child_exact_ops, 0,
         s.rq_child_exact_ns);
    emit("rq_extra_exact", s.rq_extra_exact_groups, s.rq_extra_exact_ops, 0,
         s.rq_extra_exact_ns);
    emit("rq_query_prepare", s.rq_query_prepares, 0, 0, s.rq_query_prepare_ns);
}

int main(int argc, char** argv) {
    string base_file, query_file, graph_file, qg_codebook, rabitq_table, signs_table;
    string trace_file, csv_file;
    string mode = "codebook";
    int degree = 32;
    int repeat = 1;
    int warmup = 1;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-base" && i + 1 < argc) base_file = argv[++i];
        else if (a == "-query" && i + 1 < argc) query_file = argv[++i];
        else if (a == "-graph" && i + 1 < argc) graph_file = argv[++i];
        else if (a == "-qg" && i + 1 < argc) qg_codebook = argv[++i];
        else if (a == "-rabitq" && i + 1 < argc) rabitq_table = argv[++i];
        else if (a == "-sign" && i + 1 < argc) signs_table = argv[++i];
        else if (a == "-trace" && i + 1 < argc) trace_file = argv[++i];
        else if (a == "-csv" && i + 1 < argc) csv_file = argv[++i];
        else if (a == "-mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "-degree" && i + 1 < argc) degree = atoi(argv[++i]);
        else if (a == "-repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (a == "-warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown or incomplete arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (base_file.empty() || query_file.empty() || trace_file.empty()) {
        fprintf(stderr,
                "Usage: %s -base <base.fvecs> -query <query.fvecs> -trace <trace.csv>"
                " [-graph <graph.bin>] [-mode <codebook|bins>] [-qg <qg.index>]"
                " [-rabitq <codes.bin> -sign <sign.bin>] [-degree 32]"
                " [-repeat 1] [-warmup 1] [-csv out.csv]\n",
                argv[0]);
        return 1;
    }
    if (mode != "codebook" && mode != "bins") {
        fprintf(stderr, "Error: -mode must be codebook or bins.\n");
        return 1;
    }
    if (repeat < 1) repeat = 1;
    if (warmup < 0) warmup = 0;

    g_metric_type = infer_metric_from_dataset_path(base_file);

    size_t n_base = 0, dim_base = 0, n_query = 0, dim_query = 0;
    const float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim_base);
    const float* query_vectors = read_fvecs(query_file.c_str(), n_query, dim_query);
    if (dim_base != dim_query) {
        fprintf(stderr, "Error: base dim %zu != query dim %zu.\n", dim_base, dim_query);
        return 1;
    }

    vector<AccessTraceRecord> records = read_access_trace_csv(trace_file);
    size_t active_query_count = 0;
    vector<uint8_t> seen_queries(n_query, 0);
    for (const auto& r : records) {
        if (r.query_id >= n_query) {
            fprintf(stderr, "Error: trace query_id exceeds query rows.\n");
            return 1;
        }
        if (!seen_queries[r.query_id]) {
            seen_queries[r.query_id] = 1;
            active_query_count++;
        }
    }
    sort(records.begin(), records.end(),
         [](const AccessTraceRecord& a, const AccessTraceRecord& b) {
             if (a.query_id != b.query_id) return a.query_id < b.query_id;
             if (a.step != b.step) return a.step < b.step;
             return a.op < b.op;
         });

    unique_ptr<Graph> graph;
    if (trace_has_base_op(records)) {
        if (graph_file.empty()) {
            fprintf(stderr, "Error: trace contains base ops, so -graph is required.\n");
            return 1;
        }
        graph.reset(new Graph());
        graph->load_graph_index(graph_file.c_str());
    }

    unique_ptr<QuantizationGraph> qg;
    if (trace_has_rabitq_op(records)) {
        if (mode == "codebook") {
            if (qg_codebook.empty()) {
                fprintf(stderr, "Error: RaBitQ trace requires -qg in codebook mode.\n");
                return 1;
            }
            qg.reset(new QuantizationGraph(n_base, dim_base, degree, qg_codebook));
        } else {
            if (graph_file.empty() || rabitq_table.empty() || signs_table.empty()) {
                fprintf(stderr,
                        "Error: RaBitQ bins mode requires -graph, -rabitq, and -sign.\n");
                return 1;
            }
            qg.reset(new QuantizationGraph(n_base, dim_base, degree,
                                           graph_file, rabitq_table, signs_table,
                                           base_vectors));
        }
        qg->set_metric(g_metric_type);
        degree = qg->get_degree();
    }

    printf("CPU replay trace: records=%zu queries=%zu metric=%s repeat=%d warmup=%d\n",
           records.size(), active_query_count, metric_name(g_metric_type), repeat, warmup);

    volatile float sink = 0.0f;

    auto run_once = [&](bool measure) -> ReplayStats {
        ReplayStats stats;
        stats.records = records.size();

        vector<float> appro_dist;
        vector<uint32_t> raw_buf_32;
        vector<float> result_f_buf;
        QuantizationGraph::QueryBuffers query_buffers;
        QuantizationGraph::QueryFactors qf;
        bool qf_ready = false;
        uint32_t qf_query_id = UINT32_MAX;

        if (qg) {
            const size_t padded_dim = static_cast<size_t>(qg->get_padded_dim());
            query_buffers.rotated_query =
                static_cast<float*>(aligned_alloc(64, padded_dim * sizeof(float)));
            query_buffers.quantized_query =
                static_cast<uint8_t*>(aligned_alloc(64, padded_dim * sizeof(uint8_t)));
            if constexpr (FASTSCAN_ENABLED) {
                query_buffers.lut =
                    static_cast<uint8_t*>(aligned_alloc(64, padded_dim << 2));
            }
            appro_dist.resize(degree);
            raw_buf_32.resize(degree);
            result_f_buf.resize(degree);
        }

        auto ensure_qf = [&](uint32_t query_id) {
            if (!qg || (qf_ready && qf_query_id == query_id)) {
                return;
            }
            const float* query = query_vectors + static_cast<size_t>(query_id) * dim_base;
            auto t0 = chrono::steady_clock::now();
            qf = qg->query_prepare(query, query_buffers);
            auto t1 = chrono::steady_clock::now();
            if (measure) {
                stats.rq_query_prepares++;
                stats.rq_query_prepare_ns += chrono::duration<double, nano>(t1 - t0).count();
            }
            qf_query_id = query_id;
            qf_ready = true;
        };

        for (const auto& rec : records) {
            if (rec.query_id >= n_query) {
                throw runtime_error("access trace query_id exceeds query file rows");
            }
            const float* query = query_vectors + static_cast<size_t>(rec.query_id) * dim_base;
            const uint32_t active = rec.active_mask;

            if (rec.op == ACCESS_TRACE_BASE_SEED_EXACT) {
                auto t0 = chrono::steady_clock::now();
                sink += compute_distance(dim_base, query,
                                         base_vectors + static_cast<size_t>(rec.node_id) * dim_base);
                auto t1 = chrono::steady_clock::now();
                if (measure) {
                    stats.base_exact_groups++;
                    stats.base_exact_ops++;
                    stats.base_exact_ns += chrono::duration<double, nano>(t1 - t0).count();
                }
            } else if (rec.op == ACCESS_TRACE_BASE_EXPAND_EXACT) {
                if (!graph) throw runtime_error("base trace replay needs graph");
                const vidType* neighbors = graph->get_neighbors(rec.node_id);
                uint64_t ops = 0;
                float acc = 0.0f;
                auto t0 = chrono::steady_clock::now();
                for (int rank = 0; rank < static_cast<int>(graph->degree) && rank < 32; ++rank) {
                    if ((active & (1u << rank)) == 0) continue;
                    const vidType child = neighbors[rank];
                    acc += compute_distance(dim_base, query,
                                            base_vectors + static_cast<size_t>(child) * dim_base);
                    ops++;
                }
                auto t1 = chrono::steady_clock::now();
                sink += acc;
                if (measure) {
                    stats.base_exact_groups++;
                    stats.base_exact_ops += ops;
                    stats.base_exact_ns += chrono::duration<double, nano>(t1 - t0).count();
                }
            } else if (rec.op == ACCESS_TRACE_RQ_SEED_EXACT) {
                if (!qg) throw runtime_error("RaBitQ seed trace replay needs qg");
                auto t0 = chrono::steady_clock::now();
                sink += compute_distance(dim_base, query, qg->get_row(rec.node_id));
                auto t1 = chrono::steady_clock::now();
                if (measure) {
                    stats.rq_seed_exact_groups++;
                    stats.rq_seed_exact_ops++;
                    stats.rq_seed_exact_ns += chrono::duration<double, nano>(t1 - t0).count();
                }
            } else if (rec.op == ACCESS_TRACE_RQ_EXPAND) {
                if (!qg) throw runtime_error("RaBitQ trace replay needs qg");
                ensure_qf(rec.query_id);
                float exact_dist = 0.0f;
                auto t0 = chrono::steady_clock::now();
                exact_dist = compute_distance(dim_base, query, qg->get_row(rec.node_id));
                qg->scan_all_neighbors_32(rec.node_id, exact_dist, appro_dist.data(),
                                          qf, raw_buf_32.data(), result_f_buf.data());
                auto t1 = chrono::steady_clock::now();
                sink += exact_dist + appro_dist[0];
                if (measure) {
                    stats.rq_expand_groups++;
                    stats.rq_expand_exact_ops++;
                    stats.rq_expand_est_ops += trace_popcount(active, degree);
                    stats.rq_expand_ns += chrono::duration<double, nano>(t1 - t0).count();
                }
            } else if (rec.op == ACCESS_TRACE_RQ_CHILD_EXACT ||
                       rec.op == ACCESS_TRACE_RQ_EXTRA_EXACT) {
                if (!qg) throw runtime_error("RaBitQ exact trace replay needs qg");
                const vidType* neighbors = qg->get_neighbors(rec.node_id);
                uint64_t ops = 0;
                float acc = 0.0f;
                auto t0 = chrono::steady_clock::now();
                for (int rank = 0; rank < degree && rank < 32; ++rank) {
                    if ((active & (1u << rank)) == 0) continue;
                    const vidType child = neighbors[rank];
                    acc += compute_distance(dim_base, query, qg->get_row(child));
                    ops++;
                }
                auto t1 = chrono::steady_clock::now();
                sink += acc;
                if (measure) {
                    if (rec.op == ACCESS_TRACE_RQ_CHILD_EXACT) {
                        stats.rq_child_exact_groups++;
                        stats.rq_child_exact_ops += ops;
                        stats.rq_child_exact_ns += chrono::duration<double, nano>(t1 - t0).count();
                    } else {
                        stats.rq_extra_exact_groups++;
                        stats.rq_extra_exact_ops += ops;
                        stats.rq_extra_exact_ns += chrono::duration<double, nano>(t1 - t0).count();
                    }
                }
            }
        }

        if (qg) {
            free(query_buffers.rotated_query);
            free(query_buffers.quantized_query);
            if constexpr (FASTSCAN_ENABLED) {
                free(query_buffers.lut);
            }
        }

        return stats;
    };

    for (int i = 0; i < warmup; ++i) {
        (void)run_once(false);
    }

    ReplayStats total;
    for (int i = 0; i < repeat; ++i) {
        ReplayStats cur = run_once(true);
        total.records = cur.records;
        total.base_exact_groups += cur.base_exact_groups;
        total.base_exact_ops += cur.base_exact_ops;
        total.base_exact_ns += cur.base_exact_ns;
        total.rq_expand_groups += cur.rq_expand_groups;
        total.rq_expand_exact_ops += cur.rq_expand_exact_ops;
        total.rq_expand_est_ops += cur.rq_expand_est_ops;
        total.rq_expand_ns += cur.rq_expand_ns;
        total.rq_seed_exact_groups += cur.rq_seed_exact_groups;
        total.rq_seed_exact_ops += cur.rq_seed_exact_ops;
        total.rq_seed_exact_ns += cur.rq_seed_exact_ns;
        total.rq_child_exact_groups += cur.rq_child_exact_groups;
        total.rq_child_exact_ops += cur.rq_child_exact_ops;
        total.rq_child_exact_ns += cur.rq_child_exact_ns;
        total.rq_extra_exact_groups += cur.rq_extra_exact_groups;
        total.rq_extra_exact_ops += cur.rq_extra_exact_ops;
        total.rq_extra_exact_ns += cur.rq_extra_exact_ns;
        total.rq_query_prepares += cur.rq_query_prepares;
        total.rq_query_prepare_ns += cur.rq_query_prepare_ns;
    }

    print_row("CPU_REPLAY", "base_expand_exact", total.base_exact_groups,
              total.base_exact_ops, 0, total.base_exact_ns);
    print_row("CPU_REPLAY", "rq_expand_bundle", total.rq_expand_groups,
              total.rq_expand_exact_ops, total.rq_expand_est_ops, total.rq_expand_ns);
    print_row("CPU_REPLAY", "rq_seed_exact", total.rq_seed_exact_groups,
              total.rq_seed_exact_ops, 0, total.rq_seed_exact_ns);
    print_row("CPU_REPLAY", "rq_child_exact", total.rq_child_exact_groups,
              total.rq_child_exact_ops, 0, total.rq_child_exact_ns);
    print_row("CPU_REPLAY", "rq_extra_exact", total.rq_extra_exact_groups,
              total.rq_extra_exact_ops, 0, total.rq_extra_exact_ns);
    print_row("CPU_REPLAY", "rq_query_prepare", total.rq_query_prepares,
              0, 0, total.rq_query_prepare_ns);

    if (!csv_file.empty()) {
        write_csv(csv_file, total);
    }

    (void)sink;
    return 0;
}
