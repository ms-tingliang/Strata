@echo on
if /i "%~1"=="--clean" or not exist "build_cuda13" (
    cmake -S . -B build-cuda13 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=OFF -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
    cmake --build build-cuda13 --target clean --config Debug -j 16
    cmake --build build-cuda13 --target clean --config Release -j 16
)
cmake --build build-cuda13 --target strata --target strata-dsv41 --config Release -j 16
cmake --build build-cuda13 --target strata --target strata-dsv41 --config Debug -j 16
