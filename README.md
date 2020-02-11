# Non-dominated Sorting Differential Evolution (NSDE)
[![Build Status](https://travis-ci.com/DARcorporation/nsde.svg?branch=master)](https://travis-ci.com/DARcorporation/nsde)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The Non-dominated Sorting Differential Evolution (NSDE) algorithm combines the strengths of Differential Evolution [1]
with those of the Fast and Elitist Multiobjective Genetic Algorithm NSGA-II [2], following the ideas presented in [3],
to provide an efficient and robust method for the global optimization of constrained and unconstrained, single- and 
multi-objective optimization problems. 

### Installation
NSDE is available on [PyPi](https://pypi.org/project/nsde), so it can be installed using `pip install nsde`. It can
also be installed using `python setup.py install` from the root of this repository. 

Note that several methods of NSDE are written in C++ to accelerate the code. Therefore, in order to install NSDE from 
source, a working C++ compiler is required. For Windows, this has only been tested using Visual Studio.

### Usage
To solve an optimization problem using NSDE, define a function which takes a single input argument, `x`, which 
represents the design vector, and outputs a list of objective values, `f`, and constraints, `g` (optional). For example:

```python
def unconstrained(x):
    return [x ** 2, (x - 2) ** 2]

def constrained(x):
    return sum(x * x), 1 - x
```

The first represents an unconstrained problem with two objectives. The second represents a constrained problem with a 
single objective. 

It is important to note that constraints are expected to be in the form `g(x) <= 0`. It is the user's responsibility to
transform constraints into this form.

Once formulated, problems can be solved using NSDE as follows:

```python
import nsde
opt = nsde.NSDE()
opt.init(constrained, bounds=[(-100, 100)] * 2)
opt.run()
x_opt = opt.pop[0]
f_opt = opt.fit[0]
```

In the last two lines, the optimal design vector and objective function value are retrieved from the optimizer.
As you can see, they correspond to the first elements of the optimizer's `pop` and `fit` arrays.
These are multi-dimensional arrays which store the population's design vectors and objective function values for each
individual in the population (1 row per individual).
At each new generation, these arrays are sorted such that the first rows correspond to the best individual and the last
to the worst.

For multi-objective problems, it is more useful to look at the pareto front:

```python
opt = nsde.NSDE()
opt.init(constrained, bounds=[(-100, 100)])
opt.run()
pareto = opt.fit[opt.fronts[0]]
```

When calling `.run()` on an instance of the `NSDE` class, the problem is solved until convergence or the maximum number
of generations is reached. Alternatively, it is also possible to solve problems one generation at a time by treating 
the instance of the `NSDE` class as an iterator:

```python
for generation in opt:
    print("f_opt = ", generation.fit[0])
```

### OpenMDAO
The NSDE algorithm can also be used in [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO) using the `NSDEDriver` class.

### References

1. Storn, R., and Price, K. "Differential Evolution – A Simple and Efficient Heuristic for global Optimization over
   Continuous Spaces." Journal of Global Optimization, Vol. 11, No. 4, 1997, pp. 341–359. [doi:10.1023/a:1008202821328](https://doi.org/10.1023/a:1008202821328). 
    
2. Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. “A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.”
   IEEE Transactions on Evolutionary Computation, Vol. 6, No. 2, 2002, pp. 182–197. [doi:10.1109/4235.996017](https://doi.org/10.1109/4235.996017). 
    
3. Madavan, N. K. "Multiobjective Optimization Using a Pareto Differential Evolution Approach." Proc. of IEEE Congress 
   on Evolutionary Computation. Vol. 2, 2002, pp. 1145-1150. [doi:10.1109/CEC.2002.1004404](https://doi.org/10.1109/CEC.2002.1004404).