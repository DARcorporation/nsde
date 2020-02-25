#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sorting.h"

namespace py = pybind11;

PYBIND11_MODULE(sorting, m) {
    m.doc() = R"pbdoc(
                Non-dominated and crowding distance sorting methods
                ---------------
                .. currentmodule:: sorting
                .. autosummary::
                   :toctree: _generate
                   dominates
                   nonDominatedSorting
                   crowdingDistanceSorting
            )pbdoc";

    m.def("dominates", py::overload_cast<const vector<float> &, const vector<float> &>(&dominates), R"pbdoc(
                Returns True if a dominates b.
            )pbdoc");
    m.def("dominates", py::overload_cast<const vector<float> &, const vector<float> &,
                                         const vector<float> &, const vector<float> &>(&dominates), R"pbdoc(
                Returns True if a dominates b, taking constraint violations into account.
            )pbdoc");

    m.def("nonDominatedSorting", py::overload_cast<const vector<vector<float>> &>(&nonDominatedSorting), R"pbdoc(
                Sort a set of candidate solutions into non-dominated fronts.
                Returns a list of solution indices for each front.
            )pbdoc");
    m.def("nonDominatedSorting", py::overload_cast<const vector<vector<float>> &,
                                                   size_t>(&nonDominatedSorting), R"pbdoc(
                Sort a set of candidate solutions into non-dominated fronts.
                Returns a list of solution indices for each front.
            )pbdoc");
    m.def("nonDominatedSorting", py::overload_cast<const vector<vector<float>> &,
                                                   const vector<vector<float>> &,
                                                   size_t>(&nonDominatedSorting), R"pbdoc(
                Sort a set of candidate solutions into non-dominated fronts.
                Returns a list of solution indices for each front.
            )pbdoc");


    m.def("crowdingDistanceSorting", &crowdingDistanceSorting, R"pbdoc(
                Compute the crowding distances of each solution in the given front.
            )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}