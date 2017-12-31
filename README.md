# GPT in CUDA
This is a joint research project by Shitian Ni and Shizhi Zhang.
Shitian Ni developed CUDA parallelize version of C++ GPT image matching project developed by Shizhi Zhang and Yukihiko Yamashita.

Compile by nvcc test.cu -arch=sm_60, need Nvidia GPU pascal generation for double precision atomicAdd CUDA function.

### Discussion
cudaGetSymbolAddress uses significant amount of time. 