#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <omp.h>

#include "../include/utils.h"
#include "../include/pqueue.hpp"
#include "../include/hash_filter.hpp"
#include "../include/qg.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;
float PRUNE_RATIO = 0.0f;
float APPROX_RATIO = 0.7f;
bool DEBUG_MODE = false;

SearchResult beam_cpu_rabitq_pruned(
    QuantizationGraph& qg, int K, const float* d_query, size_t dim, vidType entry_point)
{
    struct Candidate {
        vidType vid;
        float dist;
    };

    int degree = qg.get_degree();

    // initialize the hash table and frontiers for the search
    hash_filter<vidType> is_visited(BEAM_SIZE * BEAM_SIZE);
    vector<Candidate> curr_candidate;
    curr_candidate.reserve(degree);
    new_pqueue_t<vidType> frontier(BEAM_SIZE);

    // initialize the buffers for rabitq distance estimation
    vector<float> appro_dist;
    vector<uint32_t> raw_buf_32;
    vector<float> result_f_buf;
    appro_dist.resize(degree);
    raw_buf_32.resize(degree);
    result_f_buf.resize(degree);

    // counters for various statistics
    size_t dist_cmps = 0, dist_est = 0, neighbor_pruned = 0;

    // pre-allocate buffers for query preparation
    const size_t padded_dim = static_cast<size_t>(qg.get_padded_dim());
    QuantizationGraph::QueryBuffers query_buffers;
    query_buffers.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim * sizeof(float)));
    query_buffers.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim * sizeof(uint8_t)));
    query_buffers.lut = static_cast<uint8_t*>(aligned_alloc(64, padded_dim << 2));

    // define the reusable lambdas
    auto compute_exact_l2 = [&](vidType node) {
        float dist = compute_distance(dim, d_query, qg.get_row(node));
        dist_cmps++;
        return dist;
    };

    auto push_exact_candidate = [&](vidType node, float exact_dist) {
        is_visited.add(node);
        if (frontier.size() < BEAM_SIZE || exact_dist < frontier.get_tail_dist()) {
            frontier.push(node, exact_dist);
        }
    };

    const float prune_iteration_limit = static_cast<float>(BEAM_SIZE) * APPROX_RATIO;
    auto apply_ratio_pruning = [&](int iter) {
        if (PRUNE_RATIO <= 0.0f || static_cast<float>(iter) > prune_iteration_limit) return;

        size_t num_kept = static_cast<size_t>(static_cast<float>(degree) * (1.0f - PRUNE_RATIO));
        if (num_kept == 0 || curr_candidate.size() <= num_kept) return;

        neighbor_pruned += curr_candidate.size() - num_kept;
        nth_element(curr_candidate.begin(), curr_candidate.begin() + num_kept, curr_candidate.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.dist < b.dist;
            });
        curr_candidate.resize(num_kept);
    };

    // quantize the query once to avoid redundant computations
    auto qf = qg.query_prepare(d_query, query_buffers);

    // push the starting point to the frontier
    push_exact_candidate(entry_point, compute_exact_l2(entry_point));

    int iter = 0;
    while (frontier.has_unexpanded()) {
        iter++;
        curr_candidate.clear();

        // get the next unexpanded node from the frontier
        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        float cur_exact_dist = frontier.get_dist(idx);
        frontier.set_front_expanded();

        // get the neighbors of the current node
        const vidType* neighbors = qg.get_neighbors(cur_node);

        // apply pruning, if the prune ratio is set and current iteration within pruning iter limit
        const bool use_rabitq_pruning = PRUNE_RATIO > 0.0f && static_cast<float>(iter) <= prune_iteration_limit;
        if (use_rabitq_pruning) {
            // always use the LUT fastscan approach since it's more efficient
            qg.scan_all_neighbors_32(cur_node, cur_exact_dist, appro_dist.data(), qf, raw_buf_32.data(), result_f_buf.data());
            dist_est += degree;

            // collect all curr_candidate' matching score
            for (int i = 0; i < degree; i++) {
                vidType n = neighbors[i];
                if (is_visited.contains(n)) {
                    continue;
                }
                float est = appro_dist[i];
                if (!std::isfinite(est)) est = 0;
                curr_candidate.push_back({n, est});
            }

            // apply pruning
            apply_ratio_pruning(iter);

            // compute and push the exact distance of the kept curr_candidate
	        for (size_t i = 0; i < curr_candidate.size(); i++) {
                vidType n = curr_candidate[i].vid;
                if (is_visited.contains(n)) {
                    continue;
                }
                float exact_dist = compute_exact_l2(n);
                curr_candidate[i].dist = exact_dist;
                push_exact_candidate(n, exact_dist);
            }
        } else {
            // compute and push the exact distance of all neighbors without pruning
            for (int i = 0; i < degree; i++) {
                vidType n = neighbors[i];
                if (is_visited.contains(n)) {
                    continue;
                }
                float exact_dist = compute_exact_l2(n);
                curr_candidate.push_back({n, exact_dist});
                push_exact_candidate(n, exact_dist);
            }
        }

        if (DEBUG_MODE) {
            debug_print_candidates(iter, cur_node, curr_candidate);
            debug_print_beam_state(frontier);
        }
    }

    vector<int> result;
    result.reserve(K);
    // fill up the final result pool and return
    int topK = min(K, frontier.size());
    for (int i = 0; i < topK; i++) {
        result.push_back(frontier[i]);
    }
    while (static_cast<int>(result.size()) < K) {
        result.push_back(result.empty() ? 0 : result.back());
    }

    if (DEBUG_MODE) {
        debug_print_final_results(frontier);
    }

    free(query_buffers.rotated_query);
    free(query_buffers.quantized_query);
    free(query_buffers.lut);

    SearchResult res;
    res.neighbors = move(result);
    res.dist_cmps = dist_cmps;
    res.dist_est = dist_est;
    res.extra_dist_cmps = 0;
    res.iters = iter;
    res.neighbor_pruned = neighbor_pruned;
    return res;
}

RunStats run_ann_experiment_rabitq_pruned(
    QuantizationGraph& qg, int K, size_t n_base, size_t n_query, size_t dim,
    const float* base_vectors, const float* query_vectors, const int* groundtruth, size_t dim_gt
) {
    (void)n_base;
    (void)base_vectors;
    int actual_threads = 0;
    #pragma omp parallel
    {
        actual_threads = omp_get_num_threads();
    }
    cout << "OpenMP ANN search (" << actual_threads << " threads)\n";

    int64_t total_count_dc = 0, total_count_ds = 0, total_hop = 0;
    int64_t total_pruned = 0;
    const int query_count = DEBUG_MODE ? std::min<size_t>(1, n_query) : static_cast<int>(n_query);
    vector<vector<int>> all_results(query_count, vector<int>(K));

    vidType entry_point = qg.get_entry_point();

    auto start = std::chrono::high_resolution_clock::now();

    if (DEBUG_MODE) {
        cout << "[DEBUG] debug mode enabled: running only query 0\n";
    }

    #pragma omp parallel for schedule(dynamic, 1) \
        reduction(+:total_count_dc) reduction(+:total_count_ds) \
        reduction(+:total_hop) reduction(+:total_pruned)
    for (int q = 0; q < query_count; q++) {
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = beam_cpu_rabitq_pruned(qg, K, query_ptr, dim, entry_point);

        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_count_ds += result.dist_est;
        total_hop += result.iters;
        total_pruned += result.neighbor_pruned;
    }

    auto end = std::chrono::high_resolution_clock::now();

    RunStats stats;
    stats.runtime = std::chrono::duration<double>(end - start).count();
    stats.latency = stats.runtime / query_count * 1000.0;
    stats.throughput = double(query_count) / stats.runtime;
    stats.recall = compute_avg_recall(all_results, groundtruth, query_count, K, dim_gt);
    stats.total_dc = total_count_dc;
    stats.total_ds = total_count_ds;
    stats.total_extra_dc = 0;
    stats.total_full_dc = total_count_dc;
    stats.total_iter = total_hop;
    stats.total_pruned = total_pruned;

    printf("runtime: %.6f sec, recall@%d: %.6f\n", stats.runtime, K, stats.recall);
    printf("avg latency: %.6f ms/query, throughput: %.6f queries/sec\n",
           stats.latency, stats.throughput);
    cout << "total # exact distance computation: " << stats.total_dc << "\n";
    cout << "average # exact distance computation: " << stats.total_dc / query_count << "\n";
    cout << "total # distance estimation: " << stats.total_ds << "\n";
    cout << "average # distance estimation: " << stats.total_ds / query_count << "\n";
    cout << "total # iteration: " << stats.total_iter << "\n";
    cout << "average # iteration: " << stats.total_iter / query_count << "\n";
    cout << "total # pruned neighbors: " << stats.total_pruned << "\n";
    cout << "average # pruned neighbors: " << stats.total_pruned / query_count << "\n";

    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n" << argv[0]
        << " -base <base.fvecs>" << " -query <query.fvecs>" << " -gt <groundtruth.ivecs>"
        << " -degree <int>" << " [-mode <codebook|bins>]"
        << " [-qg <qg_codebook>]" << " [-graph <graph_index>]"
        << " [-rabitq <rabitq_table>]" << " [-sign <sign_table>]"
        << " [-k <int>]" << " [-beam <int>]" << " [-p_ratio <float>]" << " [-a_ratio <float>]"
        << " [-debug <on|off|1|0>]"
        << " [-csv <output.csv>]\n";
        return 1;
    }

    // required input files
    string base_file, query_file, gt_file, graph_file, rabitq_table, signs_table, qg_codebook;
    string mode = "codebook";
    string csv_file;
    int degree = 32;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "-base" && i + 1 < argc) {
            base_file = argv[++i];
        }
        else if (arg == "-query" && i + 1 < argc) {
            query_file = argv[++i];
        }
        else if (arg == "-gt" && i + 1 < argc) {
            gt_file = argv[++i];
        }
        else if (arg == "-mode" && i + 1 < argc) {
            mode = argv[++i];
        }
        else if (arg == "-graph" && i + 1 < argc) {
            graph_file = argv[++i];
        }
        else if (arg == "-degree" && i + 1 < argc) {
            degree = atoi(argv[++i]);
        }
        else if (arg == "-rabitq" && i + 1 < argc) {
            rabitq_table = argv[++i];
        }
        else if (arg == "-sign" && i + 1 < argc) {
            signs_table = argv[++i];
        }
        else if (arg == "-qg" && i + 1 < argc) {
            qg_codebook = argv[++i];
        }
        else if (arg == "-k" && i + 1 < argc) {
            K = atoi(argv[++i]);
        }
        else if (arg == "-beam" && i + 1 < argc) {
            BEAM_SIZE = atoi(argv[++i]);
        }
        else if (arg == "-p_ratio" && i + 1 < argc) {
            PRUNE_RATIO = atof(argv[++i]);
        }
        else if (arg == "-a_ratio" && i + 1 < argc) {
            APPROX_RATIO = atof(argv[++i]);
        }
        else if (arg == "-debug" && i + 1 < argc) {
            string debug_arg = argv[++i];
            if (debug_arg == "on" || debug_arg == "1" || debug_arg == "true") {
                DEBUG_MODE = true;
            }
            else if (debug_arg == "off" || debug_arg == "0" || debug_arg == "false") {
                DEBUG_MODE = false;
            }
            else {
                cerr << "Error: -debug must be one of on, off, 1, 0, true, false.\n";
                return 1;
            }
        }
        else if (arg == "-csv" && i + 1 < argc) {
            csv_file = argv[++i];
        }
        else {
            cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    if (mode != "codebook" && mode != "bins") {
        cerr << "Error: -mode must be either 'codebook' or 'bins'.\n";
        return 1;
    }

    if (base_file.empty() || query_file.empty() || gt_file.empty()) {
        cerr << "Error: missing required input files.\n";
        return 1;
    }

    if (mode == "codebook" && qg_codebook.empty()) {
        cerr << "Error: codebook mode requires -qg <qg_codebook>.\n";
        return 1;
    }

    if (mode == "bins" && (graph_file.empty() || rabitq_table.empty() || signs_table.empty())) {
        cerr << "Error: bins mode requires -graph <graph_index> -rabitq <rabitq_table> -sign <sign_table>.\n";
        return 1;
    }

    if (PRUNE_RATIO < 0.0f || PRUNE_RATIO > 1.0f) {
        cerr << "Error: -p_ratio must be between 0 and 1.\n";
        return 1;
    }

    if (APPROX_RATIO < 0.0f || APPROX_RATIO > 1.0f) {
        cerr << "Error: -a_ratio must be between 0 and 1.\n";
        return 1;
    }

    g_metric_type = infer_metric_from_dataset_path(base_file);
    cout << "Metric: " << metric_name(g_metric_type) << "\n";

    // load all input data
    size_t n_base, dim_base;
    size_t n_query, dim_query;
    size_t n_gt, dim_gt;

    const float* base_vectors = read_fvecs(base_file.c_str(), n_base, dim_base);
    const float* query_vectors = read_fvecs(query_file.c_str(), n_query, dim_query);
    const int* groundtruth = read_ivecs(gt_file.c_str(), n_gt, dim_gt);

    if (dim_base != dim_query) {
        cerr << "Error: base/query dimension mismatch.\n";
        return 1;
    }

    QuantizationGraph qg = (mode == "codebook")
        ? QuantizationGraph(n_base, dim_base, degree, qg_codebook)
        : QuantizationGraph(n_base, dim_base, degree, graph_file, rabitq_table, signs_table, base_vectors);
    qg.set_metric(g_metric_type);

    cout << "Debug mode: " << (DEBUG_MODE ? "on" : "off") << "\n";

    if (mode == "bins") {
        int actual_threads = 0;
        #pragma omp parallel
        {
            actual_threads = omp_get_num_threads();
        }
        vector<float> centroid = compute_centroid(base_vectors, n_base, dim_base, actual_threads);
        vidType entry_point = exact_nn(base_vectors, centroid.data(), n_base, dim_base, actual_threads);
        qg.set_entry_point(entry_point);
        cout << "Computed bins-mode entry point: " << entry_point << "\n";
    }

    // Run experiment
    RunStats stats = run_ann_experiment_rabitq_pruned(
        qg, K, n_base, n_query, dim_base, base_vectors, query_vectors, groundtruth, dim_gt);

    if (!csv_file.empty()) {
        append_stats_to_csv(csv_file, K, BEAM_SIZE, PRUNE_RATIO, APPROX_RATIO, stats);
    }

    delete[] base_vectors;
    delete[] query_vectors;
    delete[] groundtruth;
    return 0;
}
