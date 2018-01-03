# GPT in CUDA
CUDA parallelization for the paper:

[Affine-transformation and 2D-projection invariant k-NN classification of handwritten characters via a new matching measure](https://doi.org/10.1016/j.patcog.2015.10.002).

Calculate ``` g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3, gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2, tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2 ``` for equations (33),(34),(35),(36),(37),(38) in the paper.


## Dependencies
* Nvidia GPU generation >= PASCAL
* CUDA >= 8
(needed for double precision atomicAdd)

## Usage
```
nvcc main.cu -arch=sm_60
```
Tested on TSUBAME 3.0 .


Manipulate __1000times between 0 and 1 to get execution time and comparison between cpp version and cuda version.


### Discussion
cudaGetSymbolAddress uses significant amount of time. 
Calculation time reduced to half from C++ version developed by Shizhi Zhang for 1000 times executions.

## Reference
```
    @inproceedings{
        title={Affine-transformation and 2D-projection invariant k-NN classification of handwritten characters via a new matching measure},
        author={Yukihiko Yamashita, Toru Wakahara},
        booktitle={Pattern Recognition},
        year={2016}
    }
```

## License
MIT