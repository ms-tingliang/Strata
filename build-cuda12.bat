if not exist build-cuda12\ALL_BUILDS.vcxproj cmake -S . -B build-cuda12 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES="52;61;70;75;80;86;89;90;100;120" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
if /i "%1"=="" (set configuration=Release) else (set configuration=%1)
cmake --build build-cuda12 --target strata --config %configuration% -j
