#include <stdio.h>

//simulatneous comparisons for LACF
#define N 32
#define fsize 4000

//shared multiprocessor memory
__shared__ int shared_filter[fsize];

extern "C" __global__ void parallelLookup (int entry[fsize]) {  
    if ( threadIdx.x == 0 ) {
        int i;
        for ( i = 0 ; i < fsize ; i++ ) {
            shared_filter[i] = entry[i];
        }    
    }
}

extern "C" __global__ void parallelLookup1 (int* md5, int* sha, int* result) {
    int threadId = (threadIdx.x * gridDim.x) + blockIdx.x + 1;
    if ( (threadId > 0) && (threadId <= N) ) {
        int pos1 = md5[threadId];
        int pos2 = sha[threadId];
        if ( shared_filter[pos1] == 1 && shared_filter[pos2] == 1 )
            result[threadId] = threadId;
        else 
            result[threadId] = -1;
    }    
}