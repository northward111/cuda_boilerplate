//
// Created by jzj2 on 2021/8/25.
//
#include "kernel.cuh"
#include <cstdio>
#include <iostream>


static void HandleError( cudaError_t err, const char *file, int line )
{
    // CUDA error handeling from the "CUDA by example" book
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

namespace kernel{

__global__ void cu_dot(Eigen::Vector3d *v1, Eigen::Vector3d *v2, double *out, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        out[idx] = v1[idx].dot(v2[idx]);
    }
    return;
}

// The wrapper for the calling of the actual kernel
double dot(const std::vector<Eigen::Vector3d> & v1, const std::vector<Eigen::Vector3d> & v2)
{
    // print some device info
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "using GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM count " << devProp.multiProcessorCount << std::endl;
    std::cout << "sharedMemPerBlock " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "maxThreadsPerBlock " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor " << devProp.maxThreadsPerMultiProcessor << std::endl;

    int n = v1.size();
    double *ret = new double[n];

    // Allocate device arrays
    Eigen::Vector3d *dev_v1, *dev_v2;
    HANDLE_ERROR(cudaMalloc((void **)&dev_v1, sizeof(Eigen::Vector3d)*n));
    HANDLE_ERROR(cudaMalloc((void **)&dev_v2, sizeof(Eigen::Vector3d)*n));
    double* dev_ret;
    HANDLE_ERROR(cudaMalloc((void **)&dev_ret, sizeof(double)*n));

    // Copy to device
    HANDLE_ERROR(cudaMemcpy(dev_v1, v1.data(), sizeof(Eigen::Vector3d)*n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_v2, v2.data(), sizeof(Eigen::Vector3d)*n, cudaMemcpyHostToDevice));

    // Dot product
    cu_dot<<<(n+1023)/1024, 1024>>>(dev_v1, dev_v2, dev_ret, n);

    // Copy to host
    HANDLE_ERROR(cudaMemcpy(ret, dev_ret, sizeof(double)*n, cudaMemcpyDeviceToHost));

    // Reduction of the array
    for (int i=1; i<n; ++i)
    {
        ret[0] += ret[i];
    }

    // Return
    return ret[0];
}

}





