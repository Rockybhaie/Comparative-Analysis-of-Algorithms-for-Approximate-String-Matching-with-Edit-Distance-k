/**
 * bench.cpp — Unified benchmark harness for the CSE 317 group project.
 *
 * Approximate String Matching: find every occurrence of pattern P in text T
 * with at most k errors. Four algorithms are bundled here so that the
 * comparative-analysis notebook can drive them all through a single binary.
 *
 * CLI
 * ---
 *   bench <algo> <text_file> <pattern_file> <k>
 *
 *   <algo> ∈ { brute, sellers, pigeonhole, seed }
 *
 * Output (stdout, machine-readable)
 *   line 1: <algo>,<n>,<m>,<k>,<num_matches>,<elapsed_microseconds>
 *   line 2: comma-separated 0-based starting positions of matches
 *
 * Notes
 *   - All four algorithms report 0-based *starting* positions.
 *   - "brute" and "sellers" naturally produce ending positions in the DP
 *     formulation; we convert them to canonical starting positions by
 *     taking max(0, end - m + 1) so that comparison across algorithms is
 *     well defined. (Different algorithms may legitimately disagree on the
 *     exact starting position for a given match because edit distance is
 *     non-unique, so the notebook uses *position windows* for accuracy
 *     metrics.)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// 1. BRUTE FORCE — Recursive Branch-and-Bound (Khalid)
// =============================================================================
static void brute_recurse(const std::string& T, const std::string& P,
                          int t, int p, int err, int k,
                          int start, std::vector<int>& starts,
                          std::unordered_set<int>& seen)
{
    if (err > k) return;
    if (p == (int)P.size()) {
        if (!seen.count(start)) { seen.insert(start); starts.push_back(start); }
        return;
    }
    if (t == (int)T.size()) return;

    if (T[t] == P[p]) {
        brute_recurse(T, P, t + 1, p + 1, err, k, start, starts, seen);
    } else {
        brute_recurse(T, P, t + 1, p + 1, err + 1, k, start, starts, seen); // sub
        brute_recurse(T, P, t + 1, p,     err + 1, k, start, starts, seen); // ins
        brute_recurse(T, P, t,     p + 1, err + 1, k, start, starts, seen); // del
    }
}

std::vector<int> brute_search(const std::string& T, const std::string& P, int k) {
    std::vector<int> starts;
    std::unordered_set<int> seen;
    int n = (int)T.size();
    for (int i = 0; i < n; ++i) {
        brute_recurse(T, P, i, 0, 0, k, i, starts, seen);
    }
    std::sort(starts.begin(), starts.end());
    return starts;
}

// =============================================================================
// 2. SELLERS' DP — exact O(n·m) (Kashish)
// =============================================================================
std::vector<int> sellers_search(const std::string& T, const std::string& P, int k) {
    int n = (int)T.size(), m = (int)P.size();
    std::vector<int> ends;
    if (m == 0) return ends;
    std::vector<int> prev(m + 1), curr(m + 1);
    for (int i = 0; i <= m; ++i) prev[i] = i;
    for (int j = 1; j <= n; ++j) {
        curr[0] = 0;
        for (int i = 1; i <= m; ++i) {
            int cost = (P[i - 1] == T[j - 1]) ? 0 : 1;
            curr[i] = std::min({ prev[i - 1] + cost,
                                 prev[i]     + 1,
                                 curr[i - 1] + 1 });
        }
        if (curr[m] <= k) ends.push_back(j - 1); // 0-based end
        std::swap(prev, curr);
    }
    // Convert ending positions → canonical start = max(0, end - m + 1)
    std::vector<int> starts;
    starts.reserve(ends.size());
    for (int e : ends) starts.push_back(std::max(0, e - m + 1));
    // Dedup adjacent equal starts (DP often reports a run of consecutive ends
    // for the same match; collapse to one start per match cluster).
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    return starts;
}

// =============================================================================
// 3. PIGEONHOLE PRINCIPLE FILTER + DP verify (Muskan)
// =============================================================================
// Returns the local end-columns (0-based, within W) at which an approximate
// match of P with edit distance <= k ends.  These are the canonical "end
// positions" used by Sellers' DP, so the caller can convert them back into
// global match-start positions in T by:
//      global_end   = window_start + local_end
//      global_start = max(0, global_end - m + 1)
static std::vector<int> verify_window(const std::string& P, const std::string& W,
                                      int k) {
    std::vector<int> ends;
    int m = (int)P.size(), w = (int)W.size();
    if (w == 0) return ends;
    std::vector<int> prev(m + 1), curr(m + 1);
    for (int i = 0; i <= m; ++i) prev[i] = i;
    for (int j = 1; j <= w; ++j) {
        curr[0] = 0;
        for (int i = 1; i <= m; ++i) {
            int cost = (P[i - 1] == W[j - 1]) ? 0 : 1;
            curr[i] = std::min({ prev[i] + 1,
                                 curr[i - 1] + 1,
                                 prev[i - 1] + cost });
        }
        if (curr[m] <= k) ends.push_back(j - 1);  // 0-based local end
        std::swap(prev, curr);
    }
    return ends;
}

std::vector<int> pigeonhole_search(const std::string& T, const std::string& P, int k) {
    int n = (int)T.size(), m = (int)P.size();
    std::set<int> matched_starts;
    if (m == 0 || m > n) return {};
    int pieces = k + 1;
    int r = std::max(1, m / pieces);

    for (int idx = 0; idx < pieces; ++idx) {
        int cur_r = (idx == pieces - 1) ? (m - idx * r) : r;
        if (cur_r <= 0) continue;
        int p_off = idx * r;                                  // offset of this piece in P
        std::string block = P.substr(p_off, cur_r);
        size_t pos = T.find(block, 0);
        while (pos != std::string::npos) {
            int i = (int)pos;
            // For a piece located at offset p_off in P that hits T at position i,
            // the corresponding match in T (if any) must start within [i-p_off-k, i-p_off+k].
            // Allow up to k slack on either side; the window must therefore span:
            //      left  = i - p_off - k
            //      right = i - p_off + m - 1 + k
            int start_idx = std::max(0, i - p_off - k);
            int end_idx   = std::min(n - 1, i - p_off + m - 1 + k);
            if (start_idx > end_idx) { pos = T.find(block, pos + 1); continue; }
            std::string window = T.substr(start_idx, end_idx - start_idx + 1);
            auto local_ends = verify_window(P, window, k);
            for (int le : local_ends) {
                int global_end   = start_idx + le;            // 0-based end in T
                int global_start = std::max(0, global_end - m + 1);
                matched_starts.insert(global_start);
            }
            pos = T.find(block, pos + 1);
        }
    }
    return std::vector<int>(matched_starts.begin(), matched_starts.end());
}

// =============================================================================
// 4. SEED-AND-EXTEND — greedy / approximate (Abdullah)
// =============================================================================
static int choose_seed_length(int n, int m, int k, int sigma) {
    if (m <= 3) return std::max(2, m - 1);
    int q_log    = std::max(3, (int)std::ceil(std::log((double)n) / std::log((double)sigma)));
    int q_pigeon = std::max(2, m / (k + 2));
    int q_cap    = std::max(2, m / 2);
    return std::max(2, std::min({ q_log, q_pigeon, q_cap, m - 1 }));
}

static int alphabet_size(const std::string& T) {
    bool seen[256] = {false};
    for (unsigned char c : T) seen[c] = true;
    int s = 0;
    for (bool b : seen) if (b) ++s;
    return std::max(2, s);
}

std::vector<int> seed_and_extend_search(const std::string& T,
                                        const std::string& P, int k) {
    int n = (int)T.size(), m = (int)P.size();
    if (m == 0 || m > n) return {};
    int sigma = alphabet_size(T);
    int q = choose_seed_length(n, m, k, sigma);

    std::unordered_map<std::string, std::vector<int>> idx;
    idx.reserve((size_t)(n - q + 1) * 2);
    for (int i = 0; i + q <= n; ++i) idx[T.substr(i, q)].push_back(i);

    std::set<int> reported;
    std::vector<int> starts;
    for (int p = 0; p + q <= m; ++p) {
        auto it = idx.find(P.substr(p, q));
        if (it == idx.end()) continue;
        for (int t_hit : it->second) {
            int t_start = t_hit - p;
            if (t_start < 0 || t_start + m > n) continue;
            if (reported.count(t_start)) continue;
            int err = 0;
            for (int i = 0; i < m; ++i) {
                if (T[t_start + i] != P[i]) {
                    if (++err > k) { err = -1; break; }
                }
            }
            if (err >= 0) {
                reported.insert(t_start);
                starts.push_back(t_start);
            }
        }
    }
    std::sort(starts.begin(), starts.end());
    return starts;
}

// =============================================================================
// CLI driver
// =============================================================================
static std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open " << path << "\n";
        std::exit(2);
    }
    std::ostringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    // Trim a trailing newline if present.
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: bench <brute|sellers|pigeonhole|seed> "
                     "<text_file> <pattern_file> <k>\n";
        return 1;
    }
    std::string algo = argv[1];
    std::string T    = slurp(argv[2]);
    std::string P    = slurp(argv[3]);
    int k            = std::atoi(argv[4]);

    std::vector<int> starts;
    auto t0 = Clock::now();
    if      (algo == "brute")      starts = brute_search(T, P, k);
    else if (algo == "sellers")    starts = sellers_search(T, P, k);
    else if (algo == "pigeonhole") starts = pigeonhole_search(T, P, k);
    else if (algo == "seed")       starts = seed_and_extend_search(T, P, k);
    else { std::cerr << "Unknown algo: " << algo << "\n"; return 1; }
    auto t1 = Clock::now();
    long long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::cout << algo << "," << T.size() << "," << P.size() << ","
              << k << "," << starts.size() << "," << us << "\n";
    for (size_t i = 0; i < starts.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << starts[i];
    }
    std::cout << "\n";
    return 0;
}
