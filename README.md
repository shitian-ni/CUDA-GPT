# Accelerated GPT in CUDA

## Usage

### Compile and Execute
```
nvcc *.cpp include/*.c* -o cu.out  -arch=sm_60 --use_fast_math
```

Modify include/parameter.h Line 142 from `#define MAKETEMP 0`to `#define MAKETEMP 1` for table generations. Compile and execute.
Then change back to `#define MAKETEMP 0`.
- If the step above is not done then `**"Cannot open the file!"**` error will arise.

Tested on Tokyo Tech TSUBAME 3.0
 - Nvidia Tesla P100
 - Cuda compilation tools, release 8.0, V8.0.61

## Known issues

### minor result difference between CPU and GPU
- Could be CPU precision error in multiplyVect3x3 used in bilinear_normal_projection function in stdGpt.cpp

## Reference
```
    @inproceedings{
        title={Theoretical criterion for image matching using GPT correlation},
        author={Shizhi Zhang, Toru Wakahara, Yukihiko Yamashita},
        booktitle={2016 23rd International Conference on Pattern Recognition (ICPR)},
        year={2016},
        DOI={10.1109/ICPR.2016.7899692}
    }
    @inproceedings{
        title={Image Matching Using GPT Correlation Associated with Simplified HOG Patterns},
        author={Shizhi Zhang, Toru Wakahara, Yukihiko Yamashita},
        booktitle={2017 7th International Conference on Image Processing Theory, Tools and Applications (IPTA)},
        year={2017},
    }
``` 

## License
Apache License, Version 2.0