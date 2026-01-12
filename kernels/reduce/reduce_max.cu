// 使用 warp_shuffle 实现
// AtomicMax 不支持 float 类型，需要手动实现

#include <cuda_runtime.h>
#include <cfloat>



__device__ static float atomicMax(float* address, float val){
   int* address_as_i = (int*)address;
   int old = *address_as_i, assumed; // obtain the old value of the address
   do {
      assumed = old;
      // use atomicCAS to compare and swap the value
      old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
   } while (assumed != old); // if the value is not changed, the loop will end
   return __int_as_float(old); // return the old value
}

__global__ void max_kernel(float* input, float* output, int N){
   __shared__ float s_mem[32];

   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   int warp_id = threadIdx.x / warpSize;
   int laneId = threadIdx.x % warpSize; // the idx in the warp

   float val = (idx < N) ? input[idx] : -FLT_MAX; // store the value in register
   #pragma unroll
   for(int offset = warpSize; offset > 0; offset >>= 1){
      val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
   }
   if(laneId == 0) atomicMax(&s_mem[warp_id], val);
   __syncthreads();

   // reduce in the block, to the first warp
   if(warp_id == 0){
      int warpNum = blockDim.x / warpSize;
      val = (laneId < warpNum) ? s_mem[laneId] : -FLT_MAX;
      for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
         // use the custom atomicMax function
         val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
      }
      if(laneId == 0) atomicMax(output, val);
   }
}
