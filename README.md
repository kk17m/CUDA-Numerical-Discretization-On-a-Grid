# CUDA-Numerical-Discretization-On-a-Grid
A CUDA based numerical discretization algorithm on a rectangular grid using Gauss-Legendre quadrature rule. 

## Description

The numerical discretization algorithm is develepod for two dimensional sufficiently smooth functions. The discretization method is a projection based scheme from the continuous space into a finite dimensional representation space. The projection operator uses a pixel based expansion set to obtain the coefficients in the representation space. The computations are parallelized on a 2D rectangular grid defined by the width of the nth expansion along the respective dimension. 

* The numerical discretization algorithm is implemented in the CUDA kernel file [Numerical-Discretization-GLQ.cu](Numerical-Discretization-GLQ.cu) and the Jupyter nootbook [Numerical_Discretization_GLQ.ipynb](Numerical_Discretization_GLQ.ipynb).

* The Python 3 Jupyter notebook [Numerical_Discretization_GLQ.ipynb](Numerical_Discretization_GLQ.ipynb) can be used on the Google colaboratory cloud service that provides a free access to the NVIDIA Tesla K80 GPU.  

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk17m/kk17m/CUDA-Numerical-Discretization-On-a-Grid/blob/master/Numerical_Discretization_GLQ.ipynb)

* The nodes and weights are stored as list of lists using bidimensional arrays. The Legendre nodes (roots of the Legendre polynomials) in the interval (-1, 1) can be computed using mathematica nootbook [Nodes-weights-Gauss-Legendre.nb](Nodes-weights-Gauss-Legendre.nb). 

* The Thrust C++ tempelate libraries are used here for abstraction and performace.

### Future Extensions

* Fast implementation to directly compute the Legendre nodes and weights for the Gauss-Legendre quadrature.

* Adaptive quadrature rule to compute the error estimates.

### References

* Hildebrand, F. B. Introduction to Numerical Analysis. New York: McGraw-Hill, pp. 323-325, 1956.

* Atkinson, Kendall E. "A survey of numerical methods for solving nonlinear integral equations." The Journal of Integral Equations and Applications (1992): 15-46.

* Atkinson, Kendall E., and Florian A. Potra. "Projection and iterated projection methods for nonlinear integral equations." SIAM journal on numerical analysis 24.6 (1987): 1352-1373.

## License & Copyright
Licensed under the [MIT License](LICENSE)
