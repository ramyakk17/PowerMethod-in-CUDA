// For main look at power.cu
// GPU functions to find eigenvalues using power method

/*****************************************************************************
This function finds the product of Matrix A and vector V
*****************************************************************************/

__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int aBegin = N * BLOCK_SIZE * bx;
    int aEnd   = aBegin + N - 1;
    int step  = BLOCK_SIZE;

    int bBegin = 0;//BLOCK_SIZE * bx;
    int bIndex=0;
    int aIndex =0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += step, b += step)
    {

        __shared__ float As[BLOCK_SIZE];

        __shared__ float bs[BLOCK_SIZE];
        aIndex = a  + tx;
        if( aIndex < N*N)
        	As[tx] = g_MatA[aIndex];
		else
        	As[tx] = 0;

            bIndex = b+tx;
   	    if(bIndex<N)   
		bs[tx] = g_VecV[bIndex];
	    else
		bs[tx] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[k] * bs[k];
        }//}
        __syncthreads();
    }

    g_VecW[ BLOCK_SIZE * bx + tx] = Csub;
}

/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdata[tid] =  g_VecW[globalid];
  }
  else {
     sdata[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  sdata[tid] = sdata[tid] * sdata[tid];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_NormW,sdata[0]);
}


__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sNormData[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  if(tid==0) sNormData[0] =  g_NormW[0];
  __syncthreads();

  // For thread ids greater than data space
  if (globalid < N) {
     g_VecV[globalid] = g_VecW[globalid]/sNormData[0];
  }

}


__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdataVW[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdataVW[tid] =  g_VecV[globalid] * g_VecW[globalid];
  }
  else {
     sdataVW[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdataVW[tid] = sdataVW[tid] + sdataVW[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_Lamda,sdataVW[0]);
}




