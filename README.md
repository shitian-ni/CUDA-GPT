# Accelerated GPT in CUDA

## Usage
```
nvcc *.cpp include/*.c* -o cu.out  -arch=sm_60
```
Tested on Tokyo Tech TSUBAME 3.0
 - Nvidia Tesla P100
 - Cuda compilation tools, release 8.0, V8.0.61

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