/**
 * CPU simulation of GPU rabitq search logic (dedup-at-pick approach):
 * - topk buffer (fixed size, sorted)
 * - hash only contains expanded nodes (inserted at pick time)
 * - neighbor filtering uses hash.contains (not insert)
 * - if picked node is already in hash, skip (continue)
 *
 * Tests whether this logic can enter pathological states.
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "data_loader.hpp"
#include "cpu_graph.hpp"
#include "rabitq.hpp"

static constexpr int BQUERY = 6;

struct SimResult {
    int total_iters;
    int expansions;
    int skipped_dups;
    int found_zero;
};

SimResult simulate_gpu_logic(
    const CPUGraph& graph, const CPURabitqData& rq,
    const float* query, unsigned topk_size, vid_t entry_point, int max_iterations)
{
    int dim = graph.dim;
    int pdim = rq.padded_dim;
    int degree = graph.degree;
    int num_codebooks = pdim / 4;

    // ── prepare query (same as CPU rabitq search) ────────────────────
    std::vector<float> q_rot(pdim);
    for (int d = 0; d < dim; ++d) q_rot[d] = query[d] * rq.signs[d];
    for (int d = dim; d < pdim; ++d) q_rot[d] = 0.0f;
    for (int len = 1; len < pdim; len <<= 1)
        for (int i = 0; i < pdim; i += len << 1)
            for (int j = 0; j < len; ++j) {
                float u = q_rot[i+j], v = q_rot[i+j+len];
                q_rot[i+j] = u+v; q_rot[i+j+len] = u-v;
            }

    float low = q_rot[0], high = q_rot[0];
    for (int d = 1; d < pdim; ++d) { low = std::min(low, q_rot[d]); high = std::max(high, q_rot[d]); }
    float w = (high - low) / ((1 << BQUERY) - 1);
    float inv_w = 1.0f / w;

    std::vector<uint8_t> quant(pdim);
    int32_t sum_q = 0;
    for (int d = 0; d < pdim; ++d) {
        uint8_t qv = static_cast<uint8_t>(lroundf((q_rot[d] - low) * inv_w + 0.5f));
        quant[d] = qv; sum_q += qv;
    }

    std::vector<uint8_t> lut(num_codebooks * 16);
    for (int cb = 0; cb < num_codebooks; ++cb) {
        const uint8_t* qc = quant.data() + cb*4;
        uint8_t* lc = lut.data() + cb*16;
        lc[0] = 0;
        for (int mask = 1; mask < 16; ++mask) {
            int lb = mask & -mask;
            int pos = (lb==8)?0:(lb==4)?1:(lb==2)?2:3;
            lc[mask] = static_cast<uint8_t>(lc[mask-lb] + qc[pos]);
        }
    }

    auto l2_exact = [&](vid_t node) {
        const auto vec = graph.vect(node);
        float s = 0; for (int d = 0; d < dim; ++d) { float diff = query[d]-vec[d]; s += diff*diff; } return s;
    };

    auto estimate = [&](vid_t node, int nbr_idx, float exact_dist) -> float {
        const float* factors = rq.factors.data() + static_cast<size_t>(node)*3*degree;
        float tx = factors[nbr_idx], fdq = factors[degree+nbr_idx], fvq = factors[2*degree+nbr_idx];
        if (tx == FLT_MAX) return FLT_MAX;
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(
            rq.codes.data() + static_cast<size_t>(node)*degree*rq.bitcode_words);
        int batch = nbr_idx/32, lane = nbr_idx%32;
        const uint8_t* bp = codes + batch*pdim*4;
        uint32_t raw = 0;
        for (int cb = 0; cb < num_codebooks; ++cb) {
            int pi = cb>>1, ng = cb&1, l16 = lane&15;
            int bi = ((l16&7)<<1)|(l16>>3);
            uint8_t val = bp[pi*32+ng*16+bi];
            uint8_t code = (lane<16)?(val&0x0f):(val>>4);
            raw += lut[cb*16+code];
        }
        float rf = static_cast<float>((static_cast<int32_t>(raw)<<1)-sum_q);
        float est = tx + exact_dist + fdq*w*rf + fvq*low;
        return std::isfinite(est) ? est : FLT_MAX;
    };

    // ── simulate GPU topk + pick + dedup-at-pick ─────────────────────
    // topk: sorted vector, keep best topk_size entries
    struct Entry { float dist; vid_t node; bool marked; };
    std::vector<Entry> topk;
    topk.reserve(topk_size + degree);

    // hash: only expanded nodes
    std::unordered_set<vid_t> expanded;

    // seed entry point
    topk.push_back({l2_exact(entry_point), entry_point, false});

    auto topk_sort = [](const Entry& a, const Entry& b) { return a.dist < b.dist; };

    SimResult result = {0, 0, 0, 0};

    for (int iter = 0; iter < max_iterations; iter++) {
        result.total_iters = iter + 1;

        // pick: find best unmarked entry
        std::sort(topk.begin(), topk.end(), topk_sort);
        auto it = std::find_if(topk.begin(), topk.end(),
            [](const Entry& e) { return !e.marked; });

        if (it == topk.end()) { result.found_zero = 1; break; }

        vid_t current = it->node;
        it->marked = true;

        // dedup: try insert into expanded set
        if (!expanded.insert(current).second) {
            // duplicate — already expanded
            result.skipped_dups++;
            continue;
        }

        result.expansions++;

        // expand: estimate neighbors, filter by expanded set (contains only)
        float exact_dist = l2_exact(current);
        auto nbrs = graph.nbrs(current);
        for (int i = 0; i < degree; ++i) {
            vid_t child = nbrs[i];
            if (child == UINT32_MAX) continue;
            if (expanded.count(child)) continue;  // contains check only
            float est = estimate(current, i, exact_dist);
            topk.push_back({est, child, false});
        }

        // trim to topk_size (keep best)
        if (topk.size() > topk_size) {
            std::sort(topk.begin(), topk.end(), topk_sort);
            topk.resize(topk_size);
        }
    }

    return result;
}

static vid_t compute_entry_point(const CPUGraph& graph) {
    size_t n = graph.num_nodes; int dim = graph.dim;
    const float* data = graph.vectors.data();
    std::vector<float> centroid(dim, 0.0f);
    float inv = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; i++) { const float* r = data+i*dim; for (int d=0;d<dim;d++) centroid[d]+=r[d]*inv; }
    vid_t best=0; float bd=FLT_MAX;
    for (size_t i = 0; i < n; i++) { const float* r=data+i*dim; float d=0; for (int dd=0;dd<dim;dd++){float df=r[dd]-centroid[dd];d+=df*df;} if(d<bd){bd=d;best=static_cast<vid_t>(i);} }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <graph> <base> <query> <codebook> [topk] [max_iter]" << std::endl;
        return 1;
    }
    CPUGraph graph; graph.load(argv[1], argv[2]);
    size_t nq=0, qd=0; float* queries = read_fvecs(argv[3], nq, qd);
    CPURabitqData rq = load_rabitq_codebook(argv[4], graph.num_nodes, graph.dim, graph.degree);
    vid_t entry = (rq.entry_point != UINT32_MAX) ? rq.entry_point : compute_entry_point(graph);

    unsigned topk_size = (argc > 5) ? std::stoi(argv[5]) : 256;
    int max_iter = (argc > 6) ? std::stoi(argv[6]) : 256;

    printf("data: %zu points, dim=%d, degree=%d, entry=%u\n", graph.num_nodes, graph.dim, graph.degree, entry);
    printf("config: topk=%u, max_iter=%d, queries=%zu\n\n", topk_size, max_iter, nq);

    int worst_dups = 0, worst_query = -1;
    for (size_t q = 0; q < std::min(nq, size_t(100)); q++) {
        SimResult r = simulate_gpu_logic(graph, rq, queries + q*graph.dim, topk_size, entry, max_iter);
        if (r.skipped_dups > worst_dups) { worst_dups = r.skipped_dups; worst_query = q; }
        if (q < 5 || r.skipped_dups > 50) {
            printf("query %3zu: iters=%d expansions=%d skipped_dups=%d found_zero=%d\n",
                   q, r.total_iters, r.expansions, r.skipped_dups, r.found_zero);
        }
    }
    printf("\nworst: query %d with %d skipped duplicates\n", worst_query, worst_dups);

    delete[] queries;
    return 0;
}
