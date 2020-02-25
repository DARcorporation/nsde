#include <iostream>
#include "hv.h"
#include <vector>

using namespace std;

int main() {
    vector<vector<double>> data = {{
        {0.0, 3.0},
        {2.0, 2.0},
        {3.0, 1.0}
    }};
    vector<double> ref = {4.0, 4.0};

    int n = data.size();
    int d = data[0].size();
    vector<double> flatData;
    flatData.reserve(n * d);
    for (auto& vec : data)
        for (auto& elem : vec)
            flatData.push_back(elem);

    double hv = fpli_hv(flatData.data(), d, n, ref.data());

    cout << hv << endl;
}
