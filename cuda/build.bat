@echo off
cd /d "%~dp0"
rem Build FDK CUDA reconstruction program.
rem Requires: VS Build Tools 2022 (C++ desktop workload) + CUDA Toolkit 12.1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 (
    echo [ERROR] vcvars64.bat not found. Install VS Build Tools 2022 with C++ desktop workload first.
    exit /b 1
)
nvcc -O3 -arch=sm_89 -allow-unsupported-compiler -Xcompiler /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH fdk.cu -o fdk.exe -lcufft
if %errorlevel% neq 0 (
    echo Build failed.
    exit /b 1
)
echo Build success: fdk.exe
