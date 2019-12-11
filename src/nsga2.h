//
// Created by daniel.devries on 12/5/2019.
//

#ifndef CPP_NSGA2_H
#define CPP_NSGA2_H

#include <vector>
#include <map>
using namespace std;

bool dominates(const vector<float> &P1, const vector<float> &P2);
bool dominates(const vector<float> &P1, const vector<float> &P2, const vector<float> &C1, const vector<float> &C2);

vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P);
vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P, const vector<vector<float>> &C);
vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P, size_t nP);
vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P, const vector<vector<float>> &C, size_t nP);
vector<vector<size_t>> nonDominatedSorting(const vector<vector<float>> &P, const vector<vector<float>> &C, size_t nP, bool useC);

vector<size_t> crowdingDistanceSorting(const vector<vector<float>> &I);

#endif //CPP_NSGA2_H
