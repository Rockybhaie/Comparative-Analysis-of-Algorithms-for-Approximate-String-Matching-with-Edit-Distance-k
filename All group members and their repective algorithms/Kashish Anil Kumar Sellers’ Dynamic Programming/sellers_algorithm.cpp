/**
 * Seller's Algorithm — Approximate String Matching with Edit Distance <= k
 * CSE 317: Algorithms: Design and Analysis (Spring 2026)
 * Group: Abdullah Irfan, Abdullah Khalid, Kashish Anil Kumar, Muskan Pawan
 *
 * Based on: P. H. Sellers (1980), adapted from Wagner-Fischer framework.
 *
 * Key idea:
 *   D[i][j] = min edit distance between prefix P[0..i-1] and ANY suffix of T[0..j-1].
 *   Initialise D[0][j] = 0 for all j  →  pattern may start anywhere in T.
 *   Initialise D[i][0] = i           →  deleting all i pattern chars from an empty text prefix.
 *
 * When D[m][j] <= k, a match ending at text position j is reported.
 * Space-optimised: only two columns (prev + curr) are kept at any time → O(m).
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

struct Match {
    int end_pos;   // 1-based ending position in T
    int distance;  // edit distance of this match
};

/**
 * sellers_search
 *
 * @param T  Reference text
 * @param P  Pattern to search for
 * @param k  Maximum allowed edit distance
 * @return   Vector of Match structs (end position + distance), sorted by end_pos
 */
std::vector<Match> sellers_search(const std::string& T,
                                  const std::string& P,
                                  int k)
{
    int n = (int)T.size();
    int m = (int)P.size();

    std::vector<Match> results;

    // prev[i] = D[i][j-1],  curr[i] = D[i][j]
    // Size m+1 to hold rows 0..m
    std::vector<int> prev(m + 1), curr(m + 1);

    // --- Initialise column 0 (j = 0): D[i][0] = i ---
    for (int i = 0; i <= m; i++)
        prev[i] = i;

    // --- Fill column by column (j = 1..n) ---
    for (int j = 1; j <= n; j++) {
        // D[0][j] = 0  →  pattern can start anywhere
        curr[0] = 0;

        for (int i = 1; i <= m; i++) {
            int cost = (P[i - 1] == T[j - 1]) ? 0 : 1;  // substitution cost

            curr[i] = std::min({
                prev[i - 1] + cost,   // substitution (or match)
                prev[i]     + 1,      // deletion  (delete from pattern)
                curr[i - 1] + 1       // insertion (insert into pattern)
            });
        }

        // Check last row: D[m][j] <= k  →  match ending at position j
        if (curr[m] <= k) {
            results.push_back({j, curr[m]});
        }

        std::swap(prev, curr);
    }

    return results;
}

// ---------------------------------------------------------------------------
// Demo — mirrors the illustrative example from the project document (Section 1.2)
// ---------------------------------------------------------------------------
int main()
{
    // --- Example from the project doc ---
    std::string T = "ATCGGTA";   // n = 7
    std::string P = "ATCGTA";   // m = 6
    int k = 1;

    std::cout << "=== Seller's Algorithm: Approximate String Matching ===\n\n";
    std::cout << "Text    T : " << T << "  (n = " << T.size() << ")\n";
    std::cout << "Pattern P : " << P << "  (m = " << P.size() << ")\n";
    std::cout << "Threshold k = " << k << "\n\n";

    auto matches = sellers_search(T, P, k);

    if (matches.empty()) {
        std::cout << "No matches found within edit distance " << k << ".\n";
    } else {
        std::cout << "Matches found (1-based ending positions):\n";
        std::cout << "  End Pos | Edit Distance\n";
        std::cout << "  --------|-------------\n";
        for (const auto& m : matches) {
            std::cout << "     " << m.end_pos
                      << "    |     " << m.distance << "\n";
        }
    }

    // --- Additional test: fuzzy text search scenario ---
    std::cout << "\n--- Fuzzy Text Search Example ---\n";
    T = "the cat sat on the mat";
    P = "cot";   // 1 substitution from "cat" / "mat"
    k = 1;

    std::cout << "Text    T : \"" << T << "\"\n";
    std::cout << "Pattern P : \"" << P << "\"\n";
    std::cout << "Threshold k = " << k << "\n\n";

    matches = sellers_search(T, P, k);
    std::cout << "Matches found (1-based ending positions):\n";
    std::cout << "  End Pos | Edit Distance\n";
    std::cout << "  --------|-------------\n";
    for (const auto& m : matches) {
        std::cout << "     " << m.end_pos
                  << "    |     " << m.distance << "\n";
    }

    return 0;
}
