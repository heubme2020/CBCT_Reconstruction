@echo off
rem 编译 FDK CUDA 重建程序 (RTX 4090 = sm_89)
nvcc -O3 -arch=sm_89 fdk.cu -o fdk.exe -lcufft
if %errorlevel% neq 0 (
    echo 编译失败
    exit /b 1
)
echo 编译成功: fdk.exe
