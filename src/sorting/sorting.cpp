#include <algorithm>
#include <map>
#include <limits>
#include <numeric>
#include <cmath>
#include "sorting.h"

bool dominates(const vector<float> &P1, const vector<float> &P2) {
    // Return True if P1 dominates P2
    bool hasSmallerElement = false;
    for (size_t i = 0; i < P1.size(); i++) {
        if (P1[i] > P2[i]) return false;
        if (P1[i] < P2[i]) hasSmallerElement = true;
    }
    return hasSmallerElement;
}

bool dominates(const vector<float> &P1, const vector<float> &P2,
               const vector<float> &C1, const vector<float> &C2) {
    // Return True if P1 dominates P2, incorporating constraint violations
    float tol = 1e-6;
    float con_diff = 0.0;
    bool basedOnFitness = true;

    for (size_t i = 0; i < C1.size(); i++) {
        if (C1[i] > tol) {
            if (C2[i] > tol) {
                // Both are infeasible. Domination based on overall constraint violation.
                con_diff += C1[i] - C2[i];
                basedOnFitness = false;
            } else {
                // 1 is infeasible, but 2 is. 1 does not dominate. (2 does.)
                return false;
            }
        } else {
            if (C2[i] > tol) {
                // 1 is feasible but 2 is not. 1 dominates.
                return true;
            }
        }
    }

    if (basedOnFitness) {
        return dominates(P1, P2);
    } else {
        // Compare overall constraint violations.
        return con_diff < 0;
    }
}


vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P) {
    return nonDominatedSorting(P, vector<vector<float>>(0), (size_t)(P.size() / 2), false);
}

vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P, size_t nP) {
    return nonDominatedSorting(P, vector<vector<float>>(0), nP, false);
}

vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P, const vector<vector<float>> &C) {
    return nonDominatedSorting(P, C, (size_t)(P.size() / 2), true);
}

vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P, const vector<vector<float>> &C, size_t nP) {
    return nonDominatedSorting(P, C, nP, true);
}

vector<vector<size_t>> nonDominatedSorting(
    const vector<vector<float>> &P,
    const vector<vector<float>> &C,
    size_t nP,
    bool useC) {
    // Population size
    const size_t M = P.size();
    // Number of individuals each individual is dominated by
    map<size_t, int> n;
    // List of individuals each individual dominates
    vector<vector<size_t>> S(M);
    // List of fronts
    vector<vector<size_t>> F(1);
    // Number of solutions in the fronts
    size_t nQ = 0;

    for (size_t p = 0; p < M; p++) {
        n[p] = 0;

        for (size_t q= 0 ; q < M; q++) {
            if (p != q) {
                if (useC) {
                    if (dominates(P[p], P[q], C[p], C[q])) {        // if p dominates q
                        S[p].push_back(q);                          // add q to the set of solutions dominated by p
                    } else if (dominates(P[q], P[p], C[q], C[p])) {
                        n[p]++;                                     // increment the domination counter of p
                    }
                } else {
                    if (dominates(P[p], P[q])) {                    // if p dominates q
                        S[p].push_back(q);                          // add q to the set of solutions dominated by p
                    } else if (dominates(P[q], P[p])) {
                        n[p]++;                                     // increment the domination counter of p
                    }
                }
            }
        }

        if (n[p] == 0) {
            F[0].push_back(p);
            nQ++;
        }
    }

    // If the number of solutions in the first front exceeds nP, just return it
    if (nQ >= nP) {
        return F;
    }

    while (true) {
        vector<size_t> Q;                                           // Used to store the members of the next front

        for (auto p : F.back()) {
            for (auto q : S[p]) {
                n[q]--;
                if (n[q] == 0) {                                    // q belongs to the next front
                    Q.push_back(q);
                    nQ++;
                }
            }
        }

        // If the current front is empty, return the fronts
        if (Q.empty()) return F;
        // Add this front to the list of fronts
        F.push_back(Q);
        // If the number of selected solutions exceeds nP, return the fronts
        if (nQ >= nP) return F;
    }
}

vector<size_t> crowdingDistanceSorting(const vector<vector<float>> &I) {
    // Make list of indices
    vector<size_t> idx(I.size());
    iota(begin(idx), end(idx), 0);

    size_t l = I.size(), M = I[0].size();                           // number of solutions in I
    vector<float> I_distance(I.size(), 0);                    // initialize distance

    for (size_t m = 0; m < M; m++) {
        sort(begin(idx), end(idx),
             [&I, m](size_t i1, size_t i2) {
                 return I[i1][m] < I[i2][m];
             });                                                    // sort using each objective value

        I_distance[idx[0]] = numeric_limits<float>::max();          // so that boundary points are always selected
        I_distance[idx[l-1]] = numeric_limits<float>::max();

        for (size_t i = 1; i < l-1; i++) {                          // for all other points
            I_distance[idx[i]] += (I[idx[i+1]][m] - I[idx[i-1]][m]) / (I[idx[l-1]][m] - I[idx[0]][m]);
        }
    }

    // Sort the indices of the solutions by their crowding distances (large first)
    sort(begin(idx), end(idx),
         [&I_distance](size_t i1, size_t i2) {
             return I_distance[i1] > I_distance[i2];
         });

    // Return sorted indices
    return idx;
}

