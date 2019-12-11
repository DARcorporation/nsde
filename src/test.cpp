#include <iostream>
#include "nsga2.h"

using namespace std;

int main(){
    vector<vector<float>> pop = {{
        {0.5, 1.0},
        {0.1, 0.9},
        {0.3, 0.4},
        {0.8, 0.4},
        {0.5, 0.0},
        {0.8, 0.5},
        {0.3, 0.8},
        {0.3, 0.5},
        {0.1, 0.4},
        {0.2, 0.3},
        {0.8, 0.3},
        {0.5, 1.0},
        {1.0, 0.7},
        {0.5, 0.3},
        {0.2, 1.0},
        {0.5, 0.1},
        {0.6, 0.8},
        {0.6, 0.9},
        {0.0, 0.5},
        {0.5, 0.1},
    }};
    vector<vector<size_t>> F = nonDominatedSorting(pop);

    vector<vector<float>> Fi;
    for (const auto &i : F.back()) {
        Fi.push_back(pop[i]);
    }

    vector<size_t> cd = crowdingDistanceSorting(Fi);
    cout << "hello" << endl;
//    vector<vector<size_t>> fronts = nonDominatedSorting(pop);
//
//    for (const auto &front : fronts) {
//        for (auto ind : front) {
//            cout << ind << " ";
//        }
//        cout << endl;
//    }

}