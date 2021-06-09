#ifdef __cplusplus
extern "C++" {
#include "glife.h"
using namespace std;
}
#include <cuda.h>

// HINT: YOU CAN USE THIS METHOD FOR ERROR CHECKING
// Print error message on CUDA API or kernel launch
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
        } \
    } while (0)

#define MAX_THREAD_NUM 1024
#define MAX_THREAD_SIZE 32
#define PAD 1

__device__ bool isOoB(int tot_rows, int tot_cols, int row, int col)
{
  if(row < 0 or row >= tot_rows) return true;
  if(col < 0 or col >= tot_cols) return true;
  return false;
}

__device__ int isLive(int* grid, int index) { return (grid[index] ? LIVE : DEAD); }

__device__ bool canGoLive(int * grid, int index, int n) {
  if(isLive(grid, index) and (n == 2 or n==3)) return true;
  if(!isLive(grid, index) and n==3) return true;
  return false;
}

// TODO: YOU MAY NEED TO USE IT OR CREATE MORE
__device__ int getNeighbors(int* grid, int tot_rows, int tot_cols,
        int row, int col) {
    int new_x, new_y, numOfNeighbors=0, index;
    int dx [3] = {-1, 0, 1};
    int dy [3] = {-1, 0, 1};
    for(auto x : dx){
        for(auto y: dy){
        new_x = row + x; new_y = col + y;
        index = new_x * tot_cols + new_y;
        (x==0 and y==0) or 
        isOoB(tot_rows, tot_cols, new_x, new_y) or 
        !isLive(grid, index) ? 0 : numOfNeighbors++;
        }
    }
    return numOfNeighbors;
}

// TODO: YOU NEED TO IMPLEMENT KERNEL TO RUN ON GPU DEVICE 
__global__ void kernel(int * d_grid, int * d_temp, int tot_rows, int tot_cols)
{   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int gindex = row * tot_rows + col;
    
    if(isOoB(tot_rows, tot_cols, row, col)) return;

    d_temp[gindex] = 
        canGoLive(d_grid, gindex, getNeighbors(d_grid, tot_rows, tot_cols, row, col)) ?
        1 : 0;

    return;
}

__device__ inline int dimension_conv(int row, int col, int col_size){
    return row * col_size + col;
}

__global__ void kernel2(int * d_grid, int * d_temp, int tot_rows, int tot_cols)
{   
    const int temp_size = MAX_THREAD_SIZE + (PAD * 2);
    int start_r = blockIdx.y * blockDim.y, start_c = blockIdx.x * blockDim.x;
    int row = threadIdx.y, col = threadIdx.x;

    __shared__ int temp[temp_size * temp_size];

    int t_row = row + PAD, t_col = col + PAD;
    int g_row = row+start_r, g_col = col+start_c;

    int max_r = start_r + MAX_THREAD_SIZE < tot_rows ? 
        MAX_THREAD_SIZE : (tot_rows - 1) - start_r;
    int max_c = start_c + MAX_THREAD_SIZE < tot_rows ? 
        MAX_THREAD_SIZE : (tot_rows - 1) - start_c;
    
    if (isOoB(tot_rows, tot_cols, g_row, g_col)){
        __syncthreads();
        return;
    }

    if (row < PAD) {
        if (g_row >= PAD){
            temp[dimension_conv(t_row-1, t_col, temp_size)]
                = d_grid[dimension_conv(g_row-1, g_col, tot_rows)];
        }
        temp[dimension_conv(t_row+max_r, t_col, temp_size)]
            = d_grid[dimension_conv(g_row+max_r, g_col, tot_rows)];
    }

    if (col < PAD) {
        if (g_col >= PAD){
            temp[dimension_conv(t_row, t_col-1, temp_size)]
                = d_grid[dimension_conv(g_row, g_col-1, tot_rows)];
        }
        temp[dimension_conv(t_row, t_col+max_c, temp_size)]
            = d_grid[dimension_conv(g_row, g_col+max_c, tot_rows)];
    }

    if (row < PAD && col < PAD) {
        if (g_row >= PAD && g_col >= PAD) {
            temp[dimension_conv(t_row-1, t_col-1, temp_size)]
                = d_grid[dimension_conv(g_row-1, g_col-1, tot_rows)];
        }
        if (g_row >= PAD) {
            temp[dimension_conv(t_row-1, t_col+max_c, temp_size)]
                = d_grid[dimension_conv(g_row-1, g_col+max_c, tot_rows)];
        }
        if (g_col >= PAD) {
            temp[dimension_conv(t_row+max_r, t_col-1, temp_size)]
                = d_grid[dimension_conv(g_row+max_r, g_col-1, tot_rows)];
        }
        temp[dimension_conv(t_row+max_r, t_col+max_c, temp_size)]
            = d_grid[dimension_conv(g_row+max_r, g_col+max_c, tot_rows)];
    }
    
    temp[dimension_conv(t_row, t_col, temp_size)] 
        = d_grid[dimension_conv(g_row, g_col, tot_rows)];
    
    __syncthreads();

    d_temp[dimension_conv(g_row, g_col, tot_rows)] = 
        canGoLive(temp, dimension_conv(t_row, t_col, temp_size), 
        getNeighbors(temp, temp_size, temp_size, t_row, t_col)) ?
        1 : 0;

    return;
}


// TODO: YOU NEED TO IMPLEMENT TO PRINT THE INDEX RESULTS 
void cuda_dump()
{
    
}

// TODO: YOU NEED TO IMPLEMENT TO PRINT THE INDEX RESULTS 
void cuda_dump_index()
{
    
}

// TODO: YOU NEED TO IMPLEMENT ON CUDA VERSION
uint64_t runCUDA(int rows, int cols, int gen, 
                 GameOfLifeGrid* g_GameOfLifeGrid, int display)
{
    cudaSetDevice(0); // DO NOT CHANGE THIS LINE 

    uint64_t difft;

    // ---------- TODO: CALL CUDA API HERE ----------
    int size = rows * cols * sizeof(int);
    int * d_grid, *d_temp;
    int *h_grid = g_GameOfLifeGrid->getRowAddr(0);
    int *h_temp = g_GameOfLifeGrid->getRowAddrTemp(0);

    int t_row, t_col;

    t_row = t_col = MAX_THREAD_SIZE;

    if (rows * cols < MAX_THREAD_NUM) {
        t_row = rows;
        t_col = cols;
    } else if (rows < MAX_THREAD_SIZE) {
        t_row = rows;
        t_col = MAX_THREAD_NUM / rows;
    } else if (cols < MAX_THREAD_SIZE) {
        t_col = cols;
        t_row = MAX_THREAD_NUM / cols;
    }
    
    int n_r_b = ceil((double)rows / t_row);
    int n_c_b = ceil((double)cols / t_col);

    dim3 grid(n_r_b, n_c_b);
    dim3 block(t_row, t_col);

    cudaMalloc((void**)&d_grid, size);
	cudaMalloc((void**)&d_temp, size);

    cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, h_temp, size, cudaMemcpyHostToDevice);

    // Start timer for CUDA kernel execution
    difft = dtime_usec(0);
    // ----------  TODO: CALL KERNEL HERE  ----------
   
    while(gen--){
        kernel<<<grid, block>>>(d_grid, d_temp, rows, cols);
        cudaDeviceSynchronize();
        cudaCheckErrors("Check error");
        cudaMemcpy(d_grid, d_temp, size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost);

    // Finish timer for CUDA kernel execution
    difft = dtime_usec(difft);

    // Print the results
    if (display) {
        g_GameOfLifeGrid->dump();
        g_GameOfLifeGrid->dumpIndex();
    }
    return difft;
}
#endif
