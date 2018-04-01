# CUDA-Numerical-Discretization-On-a-Grid
A CUDA based numerical discretization algorithm on a rectangular grid using Gauss-Legendre quadrature rule. 

## Description

The numerical discretization algorithm is develepod for two dimensional sufficiently smooth functions. The discretization method is a projection based scheme from the continuous space into a finite dimensional representation space. The projection operator uses a pixel based expansion set to obtain the coefficients in the representation space. The computations are parallelized on a 2D rectangular grid defined by the width of the nth expansion along the respective dimension. 

* The Thrust C++ tempelate libraries are used here for abstraction and performace.

* The spatial convolution algorithm is implemented in the CUDA kernel file [SpatialConvolutionGHQ.cu](SpatialConvolutionGHQ.cu) and the Jupyter nootbook [Spatial_Convolution_GHQ.ipynb](Spatial_Convolution_GHQ.ipynb).

* The Python 3 Jupyter notebook [Spatial_Convolution_GHQ.ipynb](Spatial_Convolution_GHQ.ipynb) can be used on the Google colaboratory cloud service that provides a free access to the NVIDIA Tesla K80 GPU.  

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk17m/CUDA-Spatial-Convolution-Gauss-Hermite-Quadrature/blob/master/Spatial_Convolution_GHQ.ipynb)

* The nodes and weights are stored as list of lists using bidimensional arrays. The Golub–Welsch algorithm is used to compute the Hermite nodes (roots of the Hermite polynomials) in the interval (-inf, inf). The mathematica nootbook [Nodes-weights-Gauss-Hermite.nb](Nodes-weights-Gauss-Hermite.nb) is supplied in order to compute the Gauss-Hermite nodes and weights. 
*NOTE: Here the weighting function* exp(x^2) *is absorbed into the weights w_i.*  

### Future Extensions

* Fast implementation of the Golub–Welsch algorithm to directly generate the Hermite nodes and weights for the Gauss-Hermite quadrature. 

### References

* Davis, P.J., and P. Rabinowitz. 1984. Methods of Numerical Integration. Academic Press.

* Jäckel, P. 2005. “A Note on Multivariate Gauss-Hermite Quadrature.” Mimeo.

*  G. H. Golub and J. H. Welsch, Calculation of Gauss quadrature rules, Math. Comp. 23 (1969),
221–230

* Hale, N., Townsend, A.: Fast and accurate computation of Gauss–Legendre and Gauss–Jacobi quadrature nodes and weights. SISC 35, A652—A672 (2013)

## License & Copyright
Licensed under the [MIT License](LICENSE)
