#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hv.h"

namespace py = pybind11;
using namespace std;

double hv(vector<vector<double>> &data, vector<double> &ref) {
    int n = data.size();
    int d = data[0].size();
    vector<double> flatData;
    flatData.reserve(n * d);
    for (auto& vec : data)
        for (auto& elem : vec)
            flatData.push_back(elem);
    return fpli_hv(flatData.data(), d, n, ref.data());
}

PYBIND11_MODULE(hv, m) {
    m.doc() = R"pbdoc(
                Hypervolume computation method
                ---------------
                .. currentmodule:: hv
                .. autosummary::
                   :toctree: _generate
                   hv
                )pbdoc";

    m.def("hv", &hv, R"pbdoc(
                    Compute the hypervolume of a dataset.
                )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}