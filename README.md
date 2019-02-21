# HEImmune

Run with the following command:

OMP_NUM_THREADS=8 python analyze.py --selection --input=/path/to/input/file

Increase OMP_NUM_THREADS to the number of cores available.

An optional GPU implementation of the KMeans color quantization is available with the --cuda switch, this requires [libKMCUDA](https://github.com/src-d/kmcuda)

Requires:

* OpenCV 4.0
* Numpy
* OpenSlide
* Scikit-image
