/**
 * Seed-and-Extend Algorithm — Approximate String Matching with up to k errors
 * CSE 317: Algorithms: Design and Analysis (Spring 2026)
 * Group: Abdullah Irfan, Abdullah Khalid, Kashish Anil Kumar, Muskan Pawan
 *
 * Paradigm : Greedy heuristic / Approximation
 * Author   : Abdullah Irfan
 *
 * Idea
 * ----
 *   1. Build a q-gram hash index of the text T (q chosen heuristically,
 *      NOT tied to (k+1) — this deliberately breaks the pigeonhole
 *      guarantee that the divide-and-conquer algorithm relies on, turning
 *      the algorithm into a fast *approximate* method).
 *   2. Slide every q-gram of the pattern P over the index to obtain a
 *      list of candidate alignment offsets in T (these are the "seed hits").
 *   3. For every candidate offset, greedily extend the alignment with an
 *      ungapped (Hamming-style) comparison, abandoning the candidate as
 *      soon as the running mismatch count exceeds k. This early-abandon
 *      behaviour is the algorithm's "greedy" character.
 *   4. Report all alignment offsets whose Hamming distance ≤ k.
 *
 * Tradeoffs
 * ---------
 *   + Extremely fast on large texts: index lookup is amortised O(1),
 *     and most candidate windows are abandoned within the first few
 *     character comparisons.
 *   - Approximate: it only finds matches whose *Hamming* distance is ≤ k.
 *     A true edit-distance match that requires an insertion or deletion
 *     in the alignment will be missed. We measure this precision/recall
 *     gap against the exact algorithms in the comparative analysis.
 *
 * Theoretical Complexity
 * ----------------------
 *   - Index construction: O(n)        time, O(n) space.
 *   - Query (per pattern): O(m + h·m) time worst-case, where h is the
 *     number of seed hits. With a heuristic q chosen so that random q-grams
 *     occur ≈ n/σ^q times, expected query time is O(m + n·m/σ^q), which
 *     is sub-linear in n for typical alphabets.
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <cmath>

struct Match {
    int start_pos;        // 0-based starting position in T
    int hamming_distance; // mismatches in the ungapped alignment
};

/**
 * Pick a heuristic seed (q-gram) length.
 *   q ≈ max(3, ceil(log_σ n)) ensures random q-grams are mostly unique.
 *   Capped at m/(k+2) so the pattern still contains enough seeds, and
 *   strictly *shorter* than the pigeonhole D&C would use (m/(k+1)),
 *   which is what makes this algorithm approximate rather than exact.
 */
int choose_seed_length(int n, int m, int k, int sigma) {
    if (m <= 3) return std::max(2, m - 1);
    // log_sigma(n) keeps random q-grams mostly unique in T.
    int q_log    = std::max(3, (int)std::ceil(std::log((double)n) / std::log((double)sigma)));
    // Strictly shorter than the pigeonhole D&C uses (m/(k+1)),
    // so we lose the exact-match guarantee → algorithm is approximate.
    int q_pigeon = std::max(2, m / (k + 2));
    // Hard cap so the pattern still has many overlapping seeds.
    int q_cap    = std::max(2, m / 2);
    int q        = std::min({q_log, q_pigeon, q_cap, m - 1});
    return std::max(2, q);
}

/**
 * Build a q-gram → list-of-positions hash index of T.
 */
std::unordered_map<std::string, std::vector<int>>
build_qgram_index(const std::string& T, int q) {
    std::unordered_map<std::string, std::vector<int>> idx;
    int n = (int)T.size();
    if (q <= 0 || q > n) return idx;
    idx.reserve((size_t)(n - q + 1) * 2);
    for (int i = 0; i + q <= n; ++i) {
        idx[T.substr(i, q)].push_back(i);
    }
    return idx;
}

/**
 * Greedy ungapped extension at alignment offset `t_start` in T.
 * Returns Hamming distance if ≤ k; otherwise returns -1 (early-abandoned).
 */
int greedy_extend(const std::string& T, const std::string& P, int t_start, int k) {
    int n = (int)T.size();
    int m = (int)P.size();
    if (t_start < 0 || t_start + m > n) return -1;   // out of bounds → reject
    int errors = 0;
    for (int i = 0; i < m; ++i) {
        if (T[t_start + i] != P[i]) {
            if (++errors > k) return -1;             // greedy early abandon
        }
    }
    return errors;
}

/**
 * seed_and_extend_search
 *
 * @param T      Reference text
 * @param P      Pattern to search
 * @param k      Maximum allowed Hamming-distance errors
 * @param sigma  Alphabet size (used only to pick seed length, default 4
 *               for DNA; pass 26 for English text, 256 for arbitrary)
 * @return       Vector of unique Match{start_pos, hamming_distance}
 *               sorted by start_pos
 */
std::vector<Match> seed_and_extend_search(const std::string& T,
                                          const std::string& P,
                                          int k,
                                          int sigma = 4) {
    int n = (int)T.size();
    int m = (int)P.size();
    std::vector<Match> results;
    if (m == 0 || m > n) return results;

    int q = choose_seed_length(n, m, k, sigma);
    auto idx = build_qgram_index(T, q);

    std::set<int> reported;                          // dedupe by start_pos

    // Slide every q-gram of P over the index (overlapping seeds).
    for (int p = 0; p + q <= m; ++p) {
        std::string seed = P.substr(p, q);
        auto it = idx.find(seed);
        if (it == idx.end()) continue;

        for (int t_hit : it->second) {
            int t_start = t_hit - p;                 // align seed in T to seed in P
            if (reported.count(t_start)) continue;

            int dist = greedy_extend(T, P, t_start, k);
            if (dist >= 0) {
                results.push_back({t_start, dist});
                reported.insert(t_start);
            }
        }
    }

    std::sort(results.begin(), results.end(),
              [](const Match& a, const Match& b) { return a.start_pos < b.start_pos; });
    return results;
}

// ---------------------------------------------------------------------------
// Demo — mirrors the illustrative example from the project document
// ---------------------------------------------------------------------------
int main() {
    // --- Example from the project doc (DNA, sigma = 4) ---
    std::string T = "ATCGGTA";
    std::string P = "ATCGTA";
    int k = 1;

    std::cout << "=== Seed-and-Extend: Approximate String Matching ===\n\n";
    std::cout << "Text    T : " << T << "  (n = " << T.size() << ")\n";
    std::cout << "Pattern P : " << P << "  (m = " << P.size() << ")\n";
    std::cout << "Threshold k = " << k << "\n\n";

    auto matches = seed_and_extend_search(T, P, k, /*sigma=*/4);

    if (matches.empty()) {
        std::cout << "No Hamming-distance matches found within "
                  << k << " errors.\n";
        std::cout << "(Note: this algorithm is approximate; an exact match\n"
                     " requiring an indel may be missed by design.)\n";
    } else {
        std::cout << "Matches found (0-based start positions):\n";
        std::cout << "  Start | Hamming Distance\n";
        std::cout << "  ------|------------------\n";
        for (const auto& mm : matches) {
            std::cout << "    " << mm.start_pos
                      << "  |        " << mm.hamming_distance << "\n";
        }
    }

    // --- Additional test 1: fuzzy text search (substitution-only match) ---
    std::cout << "\n--- Fuzzy Text Search Example ---\n";
    T = "the quick brown fox jumps over the lazy dog";
    P = "quack";   // Hamming distance 1 from "quick" at position 4
    k = 1;

    std::cout << "Text    T : \"" << T << "\"\n";
    std::cout << "Pattern P : \"" << P << "\"\n";
    std::cout << "Threshold k = " << k << "\n\n";

    matches = seed_and_extend_search(T, P, k, /*sigma=*/26);
    if (matches.empty()) {
        std::cout << "No matches found.\n";
    } else {
        std::cout << "Matches found (0-based start positions):\n";
        std::cout << "  Start | Hamming Distance\n";
        std::cout << "  ------|------------------\n";
        for (const auto& mm : matches) {
            std::cout << "    " << mm.start_pos
                      << "  |        " << mm.hamming_distance << "\n";
        }
    }

    return 0;
}
