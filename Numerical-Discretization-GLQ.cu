// ***********************************************************************
//   Numerical discretization on a grid using Gauss-Legendre quadrature
//
// ***********************************************************************

/*
   MIT License

   Copyright (c) 2018 Kunal Kumar

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

using namespace thrust::placeholders;

//
// Definig the CLOCK for performance testing.
//
long long wall_clock_time()
{
#ifdef __linux__
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

//
// The parameters to compute the discrete centers (Xn, Yn) of the expansion
// functions Psi_n(x,y) are defined here.
// The axis limits along the x-axis are given by AXIS_MIN_X and AXIS_MAX_X, the
// axis limits along the y-axis are given by AXIS_MIN_Y and AXIS_MAX_Y.
//
// NOTE: These axis limits are not the limits of integration. The limits of
// integration are (Xn - lx/2, Xn + lx/2) and (Yn - ly/2, Yn + ly/2).
//
// The number of discrete points Xn and Yn are given by NUM_PTS_X and NUM_PTS_Y.
// These points can have different sizes and should be a multiple of the
// BLOCK_SIZE in the respective dimension.
//
#define AXIS_MIN_X   -1
#define AXIS_MAX_X    1
#define AXIS_MIN_Y   -1
#define AXIS_MAX_Y    1
#define NUM_PSI_X 256
#define NUM_PSI_Y 256

//
// The CUDA parameters are defined here.
// The BLOCK_SIZE parameter for the CUDA x-dimension can be different than the
// CUDA y-dimension.
//
// The Z_BLOCK_SIZE should be a factor of sizeof(Gy)/sizeof(Gy[0]).
//
#define BLOCK_SIZE 16
#define Z_BLOCK_SIZE 4

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

//
// Define the Gauss-Hermite nodes n_i and weights w_i for
// the two integrals. The size of Gy and Gx can be different depending on the
// required precision of the quadrature approximation.
//
__constant__ double Gy[8][2] = {{-0.960289856498,0.10122853629},{-0.796666477414,0.222381034453},
                                {-0.525532409916,0.313706645878},{-0.183434642496,0.362683783378},
                                {0.183434642496,0.362683783378},{0.525532409916,0.313706645878},
                                {0.796666477414,0.222381034453},{0.960289856498,0.10122853629}};

__constant__ double Gx[8][2] = {{-0.960289856498,0.10122853629},{-0.796666477414,0.222381034453},
                                {-0.525532409916,0.313706645878},{-0.183434642496,0.362683783378},
                                {0.183434642496,0.362683783378},{0.525532409916,0.313706645878},
                                {0.796666477414,0.222381034453},{0.960289856498,0.10122853629}};

//
// Declare the global vectors Xn, Yn, Cn, and Del here.
//
thrust::host_vector<double> Del;
thrust::host_vector<double> Xn;
thrust::host_vector<double> Yn;
thrust::host_vector<double> Cn(NUM_PSI_X * NUM_PSI_Y);

//
// Define the function f(x,y) here.
//
__device__ double Fun(double x, double y)
{
    return exp(-(pow(x,2) + pow(y,2))/0.5);
}

//
// The inner quadrature sum, with weights wx and nodes nx, is computed here.
//
__device__ double Sum(double *ptrXn, double *ptrDel, double *ny, int *idx)
{
    double a = ptrXn[*idx] - ptrDel[0]/2;
    double b = ptrXn[*idx] + ptrDel[0]/2;

    double C3 = 0.5*(b - a);
    double C4 = 0.5*(b + a);
    double nx, wx, Q1 = 0.0f;;

    int Nx = sizeof(Gx)/sizeof(Gx[0]);

    for (int k=0; k<Nx; k++)
    {
        nx = C4 + C3 * Gx[k][0];
        wx = Gx[k][1];
        Q1 += wx * Fun(nx, *ny);
    }

    return C3*Q1;
}

//
// The CUDA kernel is defined here and the outer quadrature sum, with weights
// wy and nodes ny, is computed here.
//
__global__ void Discretization_Kernel(double *ptrXn, double *ptrYn, double *ptrCn, double *ptrDel){

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idy = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int idz = blockIdx.z * Z_BLOCK_SIZE + threadIdx.z;

    double c = ptrYn[idy] - ptrDel[1]/2;
    double d = ptrYn[idy] + ptrDel[1]/2;

    double C1 = 0.5*(d - c);
    double C2 = 0.5*(d + c);
    double ny, wy;
    int stride_z = blockDim.z * gridDim.z;
    int Ny = sizeof(Gy)/sizeof(Gy[0]);

    while (idz < Ny ) {
        ny = C2 + C1 * Gy[idz][0];
        wy = C1 * Gy[idz][1];
        atomicAdd( &( ptrCn[idy * NUM_PSI_X + idx]), wy * Sum(ptrXn, ptrDel, &ny, &idx));
        idz += stride_z;
    }
}


int Kernelcall(){

    thrust::device_vector<double> d_Del = Del;
    thrust::device_vector<double> d_Xn = Xn;
    thrust::device_vector<double> d_Yn = Yn;
    thrust::device_vector<double> d_Cn = Cn;

    double * ptrDel = thrust::raw_pointer_cast(&d_Del[0]);
    double * ptrXn = thrust::raw_pointer_cast(&d_Xn[0]);
    double * ptrYn = thrust::raw_pointer_cast(&d_Yn[0]);
    double * ptrCn = thrust::raw_pointer_cast(&d_Cn[0]);

    int Ny = sizeof(Gy)/sizeof(Gy[0]);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, Z_BLOCK_SIZE);
    dim3 dimGrid((Xn.size() + dimBlock.x - 1) / dimBlock.x, (Yn.size() + dimBlock.y - 1) / dimBlock.y, (Ny + dimBlock.z - 1) / dimBlock.z);

    Discretization_Kernel<<<dimGrid, dimBlock>>>(ptrXn, ptrYn, ptrCn, ptrDel);
    thrust::copy(d_Cn.begin(), d_Cn.end(), Cn.begin());

    //
    // Constant required since <Psi_n(x,y), Psi_m(x,y)> = lx*ly*Delta_nm
    //
    double NormSquared = 1/(Del[0]* Del[1]);
    thrust::transform(Cn.begin(), Cn.end(), Cn.begin(),  NormSquared * _1 );

    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));

    //
    // Save result to a file
    //
    char buffer[32]; // The filename buffer.
    snprintf(buffer, sizeof(char) * 32, "FILE%i.txt", 0);
    std::ofstream out(buffer, std::ios_base::app);
    out.setf(std::ios::scientific);
    if( !out )
    {
        std::cout << "Couldn't open file."  << std::endl;
        return 1;
    }

    for (int i = 0; i < NUM_PSI_Y; i++) {
        for (int j = 0; j < NUM_PSI_X; j++) {
            out << Cn[i * NUM_PSI_X + j] <<',';
        }
        out <<'\n';
    }

    out.close();

    return 0;
}

//
// The main() function.
//
int main(int argc, char *argv[]){

    long long before, after;
    before = wall_clock_time();                                                                     // TIME START

    double xl = AXIS_MIN_X, xr = AXIS_MAX_X, yl = AXIS_MIN_Y, yr = AXIS_MAX_Y;
    int xpix = NUM_PSI_X, ypix = NUM_PSI_Y;

    Del.push_back((xr - xl) / xpix);
    Del.push_back((yr - yl) / ypix);

    for(int i=0; i < xpix; i++){
        Xn.push_back(xl + Del[0] * (i + 0.5));
    }

    for(int i=0; i < ypix; i++){
        Yn.push_back(yl + Del[1] * (i + 0.5));
    }

    Kernelcall();

    after = wall_clock_time();                                                                      // TIME END
    fprintf(stderr, "Process took %3.5f seconds ", ((float)(after - before))/1000000000);

    return 0;
}
