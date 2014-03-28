// 
// This was done as a part of Individual project for 
// Advanced Computer Architecture summer course
// Implements the POWER method to find eigenvalues of a matrix
// Contains CPU and GPU(refer power_gpu.cu) implementation 
// Compares the time taken by each for random matrices
// NOTE: Contains a number of timers, 
// hence code looks larger than actual and most code is in main

// Includes
#include <stdio.h>
#include <math.h>
#include <cutil_inline.h>
#include "device_functions.h"

const int BLOCK_SIZE =32;
#include "power_gpu.cu"

// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* d_NormW = NULL;

// Variables to change
int GlobalSize = 50000;
int BlockSize = 32;
const float EPS = 0.000001;

// create and start timer as unsigned integer
unsigned int timer_mem = 0;
unsigned int timer_total = 0;
unsigned int timer_GPU = 0;
unsigned int timer_CPU=0;

unsigned int timer_Av = 0;
unsigned int timer_Norm = 0;
unsigned int timer_Lamda=0;


// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
void PrintArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);

// Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N);
__global__ void NormalizeW(float* g_VecV,float* g_VecW, int N);
__global__ void ComputeLamda( float* g_VecV,float* g_VecW, float * g_Lamda,int N);


void CPU_AvProduct()
{
	int N = GlobalSize;
	int matIndex =0;
    for(int i=0;i<N;i++)
	{
		h_VecW[i] = 0;
		for(int j=0;j<N;j++)
		{
			matIndex = i*N + j;
			h_VecW[i] += h_MatA[matIndex] * h_VecV[j];
			
		}
	}
}

void CPU_NormalizeW()
{
	int N = GlobalSize;
	float normW=0;
	for(int i=0;i<N;i++)
		normW += h_VecW[i] * h_VecW[i];
	
	normW = sqrt(normW);
	for(int i=0;i<N;i++)
		h_VecV[i] = h_VecW[i]/normW;
}

float CPU_ComputeLamda()
{
	int N = GlobalSize;
	float lamda =0;
	for(int i=0;i<N;i++)
		lamda += h_VecV[i] * h_VecW[i];
	
	return lamda;
}

void RunCPUPowerMethod()
{
	printf("*************************************\n");
	float oldLamda =0;
	float lamda=0;
	
	//AvProduct
	CPU_AvProduct();
	
	//power loop
	for (int i=0;i<100;i++)
	{
		CPU_NormalizeW();
		CPU_AvProduct();
		lamda= CPU_ComputeLamda();
		printf("CPU lamda at %d: %f \n", i, lamda);
		// If residual is lass than epsilon break
		if(abs(oldLamda - lamda) < EPS)
			break;
		oldLamda = lamda;	
	
	}
	printf("*************************************\n");
	
}

// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);
		
    int N = GlobalSize;
    printf("Matrix size %d X %d \n", N, N);
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t norm_size = sizeof(float);
    //float CPU_result = 0.0, GPU_result = 0.0;

    // Allocate input matrix in host memory
    h_MatA = (float*)malloc(mat_size);
    if (h_MatA == 0) 
      Cleanup();

    // Allocate initial vector V in host memory
    h_VecV = (float*)malloc(vec_size);
    if (h_VecV == 0) 
      Cleanup();

    // Allocate W vector for computations
    h_VecW = (float*)malloc(vec_size);
    if (h_VecW == 0) 
      Cleanup();

   float*  h_NormW = (float*)malloc(norm_size);
   
   //timer operations
   
   cutilCheckError(cutCreateTimer(&timer_total));
   cutilCheckError(cutCreateTimer(&timer_GPU));
   cutilCheckError(cutCreateTimer(&timer_mem));
   cutilCheckError(cutCreateTimer(&timer_CPU));

    // Initialize input matrix
    UploadArray(h_MatA, N);
    InitOne(h_VecV,N);
	
    cutilCheckError(cutStartTimer(timer_CPU));
    RunCPUPowerMethod();
    cutilCheckError(cutStopTimer(timer_CPU));
	
    // Initialize input matrix
    InitOne(h_VecV,N);

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate matrix and vectors in device memory
    cutilSafeCall( cudaMalloc((void**)&d_MatA, mat_size) );
    cutilSafeCall( cudaMalloc((void**)&d_VecV, vec_size) );
    cutilSafeCall( cudaMalloc((void**)&d_VecW, vec_size) ); // This vector is only used by the device
    cutilSafeCall( cudaMalloc((void**)&d_NormW, norm_size) ); 

    cutilCheckError(cutStartTimer(timer_total));
    cutilCheckError(cutStartTimer(timer_mem));

    //Copy from host memory to device memory
    cutilSafeCall( cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice) );
	cutilCheckError(cutStopTimer(timer_mem));
	
   //Power method loops
    float OldLamda =0;
    //First find w vector
    cutilCheckError(cutStartTimer(timer_GPU));
    Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
    cutilSafeCall( cudaThreadSynchronize() ); //Needed
    cutilCheckError(cutStopTimer(timer_GPU));
    cutilCheckMsg("kernel launch failure");
	
	
    for(int i =0;i<100;i++) //Sample Maximum iteration 100
    {
       	h_NormW[0] = 0;
	 
	cutilCheckError(cutStartTimer(timer_mem));
     	cutilSafeCall( cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice) ); //One
	cutilCheckError(cutStopTimer(timer_mem));
	 
	 //Find the 2 norm of vector w in GPU
	cutilCheckError(cutStartTimer(timer_GPU));
     	FindNormW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecW,d_NormW, N);
	cutilSafeCall( cudaThreadSynchronize() ); //Needed
	cutilCheckError(cutStopTimer(timer_GPU));
     	cutilCheckMsg("kernel launch failure");
	 
	 //Check new method
	cutilCheckError(cutStartTimer(timer_mem));
	cutilSafeCall( cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost) ); //two
	cutilCheckError(cutStopTimer(timer_mem));
	 
	h_NormW[0] = sqrt(h_NormW[0]);
	 
	cutilCheckError(cutStartTimer(timer_mem));
     	cutilSafeCall( cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice) ); //three
	cutilCheckError(cutStopTimer(timer_mem));
	 
	 //Normalize w vector in GPU and write the result to v vector
	cutilCheckError(cutStartTimer(timer_GPU));
     	NormalizeW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecW, d_NormW , d_VecV, N);
	cutilSafeCall( cudaThreadSynchronize() );
	cutilCheckError(cutStopTimer(timer_GPU));
	cutilCheckMsg("kernel launch failure");

	 //Find new w vector as Av
	cutilCheckError(cutStartTimer(timer_GPU));
     	Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
     	cutilSafeCall( cudaThreadSynchronize() ); 
	cutilCheckError(cutStopTimer(timer_GPU));
	cutilCheckMsg("kernel launch failure");

	 //reset Lamda to zero
	h_NormW[0] = 0;
	 //Copy to device
	cutilCheckError(cutStartTimer(timer_mem));
     	cutilSafeCall( cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice) ); //four
	cutilCheckError(cutStopTimer(timer_mem));
	 
     	//Compute Lamda in GPU    
	cutilCheckError(cutStartTimer(timer_GPU));
	ComputeLamda<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecV, d_VecW, d_NormW , N);
     	cutilSafeCall( cudaThreadSynchronize() ); 
	cutilCheckError(cutStopTimer(timer_GPU));
	cutilCheckMsg("kernel launch failure");

	//printf("New Lamda at %d:",i );
	cutilCheckError(cutStartTimer(timer_mem));
	cutilSafeCall( cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost) ); // five
	cutilCheckError(cutStopTimer(timer_mem));
	 
	PrintArray(h_NormW, 1);

	 // If residual is less than epsilon break
	 if(abs(OldLamda - h_NormW[0]) < EPS)
		break;
	 OldLamda = h_NormW[0];
    }

    cutilCheckError(cutStopTimer(timer_total));

    printf("Memory Transfer Time: %f (ms) \n", cutGetTimerValue(timer_mem));
    printf("GPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_GPU));
    printf("Overall Execution Time (Memory + GPU): %f (ms) \n", cutGetTimerValue(timer_total));
    printf("Overall CPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_CPU));

    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_MatA)
        cudaFree(d_MatA);
    if (d_VecV)
        cudaFree(d_VecV);
    if (d_VecW)
        cudaFree(d_VecW);
	if (d_NormW)
		cudaFree(d_NormW);
		
    // Free host memory
    if (h_MatA)
        free(h_MatA);
    if (h_VecV)
        free(h_VecV);
    if (h_VecW)
        free(h_VecW);
     if (h_NormW)
        free(h_NormW);
		
	    // Destroy (Free) timer   
    cutilCheckError(cutDeleteTimer(timer_mem));
    cutilCheckError(cutDeleteTimer(timer_total));
    cutilCheckError(cutDeleteTimer(timer_GPU));
    cutilCheckError(cutDeleteTimer(timer_CPU));
	
    cutilSafeCall( cudaThreadExit() );
    
    exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
	data[0]=1;
}

void UploadArray(float* data, int n)
{
   int total = n*n;
   int value=1;
    for (int i = 0; i < total; i++){
    	data[i] = (int) (rand() % (int)(101));//1;//value;
	value ++; if(value>n) value =1;
    }
}
void PrintArray(float* data, int n)
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }
        if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0) {
                  BlockSize = atoi(argv[i+1]);
		  i = i + 1;
	}
    }
}
