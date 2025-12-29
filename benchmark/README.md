# Benchmarking

All benchmarks were conducted using the versions of the compared methods that were available at the time (2025-12) of our experiments.
For transparency and reproducibility, we report the exact commit hashes used in our experiments.

## [CropAR_Net](https://github.com/Zhoushuchang-lab/CropARNet)

Some scripts from CropARNet were copied or adapted into our project repository.
Therefore, you do not need to clone the original repository to run our code.
The following information is provided solely for reference and reproducibility.

```sh
git clone https://github.com/Zhoushuchang-lab/CropARNet.git
cd CropARNet
git rev-parse HEAD
# Commit used in our experiments:
# d53f381de0b453d6ce626e70f0a8b1c2d0c7efde

# (Optional) To exactly reproduce our setup:
git checkout d53f381de0b453d6ce626e70f0a8b1c2d0c7efde
```

## [DNNGP](https://github.com/AIBreeding/DNNGP)
```sh
git clone https://github.com/AIBreeding/DNNGP.git
cd DNNGP
git rev-parse HEAD
# Commit used in our experiments:
# 3bbac096969fb2b46958a672d342297cb4457116

# (Optional) To reproduce the exact version:
git checkout 3bbac096969fb2b46958a672d342297cb4457116
```