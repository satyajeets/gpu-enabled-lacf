#include <stdio.h>

//simulatneous comparisons for LACF
#define N 32
#define fsize 4000

typedef struct {
    int size;
    char fingerprint[5];
}
FilterEntry;

//global GPU memory. pulled into cpu later. final ans
__device__ int longestPrefix;

//all threads please just write to this OK!
__device__ int lp[33] = {0};

//shared multiprocessor memory
__shared__ FilterEntry shared_filter[fsize];

extern "C" __global__ void parallelLookup (FilterEntry *entry) {
    int thr;

    thr = threadIdx.x;

    //thread 0 for each MP copies filter from global GPU memory 
    //to MP's shared memory
    if ( thr == 0 ) {
        int i;
        for ( i = 0 ; i < fsize ; i++ ) {
            shared_filter[i] = entry[i];
        }
    }
}

//get prefix len from threadId
__device__ int popular(int prefixLength) {
    if(prefixLength >= 14 && prefixLength <= 24) {
        return 1;
    } else {
        return 0;
    }
}

__device__ int position1(int ip) {
    ip = ip % fsize;
    if(ip < 0) {
        ip = ip * -1;
    }
    return ip;
}

__device__ int position2(int pos1, int mappedValue) {
    mappedValue = mappedValue % fsize;
    if(mappedValue < 0) {
        mappedValue = mappedValue * -1;
    }
    
    int pos2 = mappedValue ^ pos1;
    pos2 = pos2 % fsize;
    return pos2;
}

__device__ int search(char fingerprint, int pos) {
    int i;
    for(i = 0; i < 5; i++) {
        if(shared_filter[pos].fingerprint[i] == fingerprint) {            
            return 1;
        }
    }
    return 0;
}

__device__ int lookupPop(char fp, int threadId, int* md5Ip, int* md5Fp) {
    int pos1, pos2, found = 0;
    pos1 = position1(md5Ip[threadId]);
    pos2 = position2(pos1, md5Fp[threadId]);

    //search pos 1
    found = search(fp, pos1);
    if ( found != 0 ) {        
        return 1;
    }
    else { //search pos 2
        found = search(fp, pos2);
        if ( found != 0 ) {
            return 1;
        }
    }
    return 0;
}

__device__ int lookupNonPop(char fp, int threadId, int* shaIp, int* shaFp) {
    int pos1, pos2, found = 0;
    pos1 = position1(shaIp[threadId]);
    pos2 = position2(pos1, shaFp[threadId]);

    //search pos 1
    found = search(fp, pos1);
    if ( found != 0 ) {
        return 1;
    }
    else { //search pos 2
        found = search(fp, pos2);
        if ( found != 0 ) {
            return 1;
        }
    }
    return 0;
}

extern "C" __global__ void parallelLookup1 (int* md5Ip, int* shaIp, int* md5Fp, int* shaFp, int* result) {
    int threadId = (threadIdx.x * gridDim.x) + blockIdx.x + 1;
    if ( (threadId > 0) && (threadId <= N) ) {
        int isPopular = popular(threadId);
        char fp = (md5Ip[threadId] & 0xFF) % 127;
        
        if ( isPopular ) {
            if ( lookupPop(fp, threadId, md5Ip, md5Fp) )
                result[threadId] = threadId;
            else 
                result[threadId] = -1;
        } else {
            if ( lookupPop(fp, threadId, md5Ip, md5Fp) ) {
                if ( lookupNonPop(fp, threadId, shaIp, shaFp) )
                    result[threadId] = threadId;
                else 
                    result[threadId] = -1;
            } else {
                result[threadId] = -1;
            }
        }
    }    
}