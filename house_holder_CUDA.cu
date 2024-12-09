
#include <iostream>
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif


#define ROW_PER_BLOCK 8
#define COL_PER_BLOCK 128

// CUDA error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile double* sdata, int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__device__ void reduceSum(double* sdata)
{
    int tid = threadIdx.x;

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
}
__global__ void updatePivotElement(double* matrix, int lpivot,int m, int n,double *up)
{
    __shared__ double shared_mem[512];
    shared_mem[threadIdx.x] = 0.0;
    __shared__ double sumOfSquares;
    sumOfSquares = 0;
    __syncthreads();

    int iterations = (n + blockDim.x - 1) / blockDim.x; // Calculate the number of iterations needed for the row

    for (int it = 0; it < iterations; it++)
    {
        int col = it * blockDim.x + threadIdx.x + lpivot-1;
        if (col < n)
        {
            double value = matrix[(lpivot - 1) * n + col];
            shared_mem[threadIdx.x] = value * value;
        }
        __syncthreads();

        reduceSum<512>(shared_mem);
         __syncthreads();

        if (threadIdx.x == 0)
        {
            sumOfSquares += shared_mem[0];
        }

        shared_mem[threadIdx.x] = 0;
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        double cl = sqrt(sumOfSquares);
        if(matrix[(lpivot - 1) * n + lpivot - 1] > 0){cl = -cl;}
        up[lpivot-1] = matrix[(lpivot - 1) * n + lpivot - 1] - cl;
        matrix[(lpivot - 1) * n + lpivot - 1] = cl;
    }
}


__global__ void updateRowNext(double* matrix,int lpivot, int m, int n,double *up)
{
    double b = up[lpivot-1] * matrix[(lpivot - 1) * n + lpivot - 1];
    if (b >= 0.0)
        return;
    b = 1 / b;

    __shared__ double sharedPivot[512];
    __shared__ double sharedResults[512];
    __shared__ double sharedResultsAcc;

    sharedResults[threadIdx.x] =0;
    sharedResultsAcc = 0;

    int row = lpivot;

    if (row >= m)
        return;

    const int iterations = (n - lpivot + 512 - 1) / 512;

    #pragma unroll
    for (int it = 0; it < iterations; it++)
    {
        int colStart = lpivot + it * blockDim.x;

        if (colStart + threadIdx.x < n)
        {
            sharedPivot[threadIdx.x] = matrix[(lpivot - 1) * n + colStart + threadIdx.x];
        }

        if (row < m && colStart + threadIdx.x < n) {
            sharedResults[threadIdx.x] = matrix[row * n + colStart + threadIdx.x] * sharedPivot[threadIdx.x];
        } else {
            sharedResults[threadIdx.x] = 0;
        }
        __syncthreads();

        reduceSum<512>(sharedResults);
         __syncthreads();

        if (row < m && threadIdx.x == 0) {
            sharedResultsAcc += sharedResults[0];
        }
        sharedResults[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sharedResultsAcc += matrix[row * n + lpivot - 1] * up[lpivot-1];
        if (sharedResultsAcc == 0)
            return;

        sharedResultsAcc *= b;
        matrix[row * n + lpivot - 1] += sharedResultsAcc * up[lpivot-1];
    }
    __syncthreads();

    #pragma unroll
    for (int it = 0; it < iterations; it++)
    {
        int col = it * blockDim.x + threadIdx.x + lpivot;
        if (col < n && row < m)
        {
            matrix[row * n + col] += sharedResultsAcc * matrix[(lpivot - 1) * n + col];
        }
    }
}

__global__ void updateMatrix(double* matrix,int lpivot, int m, int n,double *up)
{
    double b = up[lpivot-1] * matrix[(lpivot - 1) * n + lpivot - 1];
    if (b >= 0.0)
        return;
    b = 1 / b;

    __shared__ double sharedPivot[COL_PER_BLOCK];
    __shared__ double sharedResults[ROW_PER_BLOCK][COL_PER_BLOCK];
    __shared__ double sharedResultsAcc[ROW_PER_BLOCK];

    sharedResults[threadIdx.y][threadIdx.x] =0;
    sharedResultsAcc[threadIdx.y] = 0;


    int row = lpivot + blockIdx.y * blockDim.y + threadIdx.y +1;

    if (row >= m)
        return;

    const int iterations = (n - lpivot + COL_PER_BLOCK - 1) / COL_PER_BLOCK;

    #pragma unroll
    for (int it = 0; it < iterations; it++)
    {
        int colStart = lpivot + it * blockDim.x;

        if (threadIdx.y == 0)
        {
            if (colStart + threadIdx.x < n) {
                sharedPivot[threadIdx.x] = matrix[(lpivot - 1) * n + colStart + threadIdx.x];
            }
        }
        __syncthreads();

        if (row < m && colStart + threadIdx.x < n) {
            sharedResults[threadIdx.y][threadIdx.x] = matrix[row * n + colStart + threadIdx.x] * sharedPivot[threadIdx.x];
        } else {
            sharedResults[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();
        reduceSum<COL_PER_BLOCK>(sharedResults[threadIdx.y]);
         __syncthreads();

        if (row < m && threadIdx.x == 0) {
            sharedResultsAcc[threadIdx.y] += sharedResults[threadIdx.y][0];
        }
        sharedResults[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        sharedResultsAcc[threadIdx.y] += matrix[row * n + lpivot - 1] * up[lpivot-1];
        if (sharedResultsAcc[threadIdx.y] == 0)
            return;

        sharedResultsAcc[threadIdx.y] *= b;
        matrix[row * n + lpivot - 1] += sharedResultsAcc[threadIdx.y] * up[lpivot-1];
    }
    __syncthreads();

    #pragma unroll
    for (int it = 0; it < iterations; it++)
    {
        int col = it * blockDim.x + threadIdx.x + lpivot;
        if (col < n && row < m)
        {
            matrix[row * n + col] += sharedResultsAcc[threadIdx.y] * matrix[(lpivot - 1) * n + col];
        }
    }
}

void householder(std::vector<std::vector<double>> input_matrix, const char* output_file) {
    std::size_t m = input_matrix.size();
    std::size_t n = input_matrix[0].size();

    double* temp_matrix = new double[m * n];
    double* temp_result = new double[m];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            temp_matrix[i * n + j] = input_matrix[i][j];
        }
    }

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    cudaStream_t rowStream , otherRowStream;
    cudaEvent_t rowEvent , otherRowEvent;

    gpuErrchk(cudaStreamCreate(&rowStream));
    gpuErrchk(cudaStreamCreate(&otherRowStream));
    gpuErrchk(cudaEventCreate(&rowEvent));
    gpuErrchk(cudaEventCreate(&otherRowEvent));

    double* matrix;
    gpuErrchk(cudaMalloc((void**)&matrix, sizeof(double) * m * n));
    gpuErrchk(cudaMemcpy(matrix, temp_matrix, sizeof(double) * m * n, cudaMemcpyHostToDevice));
    double* d_up;
    gpuErrchk(cudaMalloc((void**)&d_up, m * sizeof(double)));


    dim3 blockDim(COL_PER_BLOCK, ROW_PER_BLOCK);
    gpuErrchk(cudaEventRecord(start));

 updatePivotElement<<<1, 512>>>(matrix, 1, m, n,d_up);
 gpuErrchk(cudaDeviceSynchronize());

    for (int lpivot = 1; lpivot <= m; lpivot++)
    {
        if(lpivot == 1)
        {
            updateRowNext<<<1, 512, 0, rowStream>>>(matrix, lpivot, m, n, d_up);
            dim3 gridDim(1, ((m - lpivot) + ROW_PER_BLOCK - 1) / ROW_PER_BLOCK);
            updateMatrix<<<gridDim, blockDim, 0, otherRowStream>>>(matrix, lpivot, m, n, d_up);
            cudaEventRecord(otherRowEvent, otherRowStream);
            updatePivotElement<<<1, 512, 0, rowStream>>>(matrix, lpivot + 1, m, n, d_up);
            cudaEventRecord(rowEvent,rowStream);
        }
        else
        {
            cudaStreamWaitEvent(rowStream,otherRowEvent,0);
            updateRowNext<<<1, 512, 0, rowStream>>>(matrix, lpivot, m, n, d_up);
            cudaStreamWaitEvent(otherRowStream , rowEvent,0);
            dim3 gridDim(1, ((m - lpivot) + ROW_PER_BLOCK - 1) / ROW_PER_BLOCK);
            updateMatrix<<<gridDim, blockDim, 0, otherRowStream>>>(matrix, lpivot, m, n, d_up);
            cudaEventRecord(otherRowEvent, otherRowStream);
            updatePivotElement<<<1, 512, 0, rowStream>>>(matrix, lpivot + 1, m, n, d_up);
            cudaEventRecord(rowEvent,rowStream);
        }   
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuErrchk(cudaStreamSynchronize(rowStream));
    gpuErrchk(cudaStreamSynchronize(otherRowStream));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

    gpuErrchk(cudaMemcpy(temp_matrix, matrix, sizeof(double) * m * n, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::ofstream outFile(output_file);

    if (!outFile.is_open()) {
        std::cout << "Unable to open output file.";
        delete[] temp_matrix;
        return;
    }

    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (j == n - 1)
                outFile << temp_matrix[i * n + j];
            else
                outFile << temp_matrix[i * n + j] << ' ';
        }
        outFile << '\n';
    }

    outFile.close();

       // Clean up
    cudaFree(matrix);
    cudaFree(d_up);
    delete[] temp_matrix;
    cudaStreamDestroy(rowStream);
    cudaStreamDestroy(otherRowStream);
    cudaEventDestroy(rowEvent);
    cudaEventDestroy(otherRowEvent);
}

int main() {
    const char* infile = "testcase/matrix_16.txt";
    const char* output_file = "par_output.txt";

    std::ifstream input_file(infile);

    if (!input_file.is_open()) {
        std::cout << "Unable to open input file.";
        return 0;
    }

    std::vector<std::vector<double>> input_matrix;
    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        input_matrix.push_back(row);
    }
    input_file.close();
    householder(input_matrix, output_file);

    return 0;
}