// FDK cone-beam CT reconstruction (CUDA)
// Steps: cosine weighting -> ramp filter (optional, CUFFT) -> weighted backprojection
// Geometry (same as python/fdk_ref.py):
//   t = x*cos(b) + y*sin(b),  s = -x*sin(b) + y*cos(b),  U = R - t
//   u_det = D*s/U,  v_det = D*z/U,  pixel = u_det/det_pitch + u0
//
// Usage:
//   fdk.exe <proj.bin> <vol.bin> <F> <H> <W> <N> <R> <D> <det_pitch> <u0> <v0> <vol_pitch> <span_deg> <filter>
//   filter: 0=no filter (weighted backprojection), 1=R-L ramp filter
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CUFFT_CHECK(call) do { cufftResult e = (call); if (e != CUFFT_SUCCESS) { \
    fprintf(stderr, "CUFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)e); exit(1); } } while (0)

// ---- 1. cosine weighting: p' = p * D / sqrt(D^2 + u^2 + v^2) ----
__global__ void weight_kernel(const float* __restrict__ proj, float* __restrict__ out,
                              int total, int H, int W, float D, float det_pitch, float u0, float v0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w = idx % W;
    int h = (idx / W) % H;
    float u = (w - u0) * det_pitch;
    float v = (h - v0) * det_pitch;
    float wgt = D / sqrtf(D * D + u * u + v * v);
    out[idx] = proj[idx] * wgt;
}

// ---- 3. weighted backprojection: vol += p'(beta,u,v) * R^2/U^2 ----
__global__ void backproject_kernel(const float* __restrict__ proj, float* __restrict__ vol,
                                   int F, int H, int W, int N,
                                   const float* __restrict__ cosb, const float* __restrict__ sinb,
                                   float R, float D, float det_pitch, float u0, float v0, float vol_pitch) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= N || iy >= N || iz >= N) return;
    float x = (ix - (N - 1) / 2.0f) * vol_pitch;
    float y = (iy - (N - 1) / 2.0f) * vol_pitch;
    float z = (iz - (N - 1) / 2.0f) * vol_pitch;
    float sum = 0.0f;
    for (int f = 0; f < F; f++) {
        float cb = cosb[f], sb = sinb[f];
        float t = x * cb + y * sb;
        float s = -x * sb + y * cb;
        float U = R - t;
        float u_det = D * s / U;
        float v_det = D * z / U;
        float u_px = u_det / det_pitch + u0;
        float v_px = v_det / det_pitch + v0;
        int ui = (int)floorf(u_px);
        int vi = (int)floorf(v_px);
        float du = u_px - ui;
        float dv = v_px - vi;
        float val = 0.0f;
        if (ui >= 0 && ui + 1 < W && vi >= 0 && vi + 1 < H) {
            const float* row0 = proj + (size_t)f * H * W + (size_t)vi * W + ui;
            const float* row1 = row0 + W;
            float v00 = row0[0], v01 = row0[1], v10 = row1[0], v11 = row1[1];
            val = v00 * (1 - du) * (1 - dv) + v01 * du * (1 - dv) + v10 * (1 - du) * dv + v11 * du * dv;
        }
        sum += val * (R * R / (U * U));
    }
    vol[((size_t)iz * N + iy) * N + ix] = sum;
}

// ---- ramp filter helper kernels ----
__global__ void pad_kernel(const float* __restrict__ proj, cufftComplex* __restrict__ buf,
                           int proj_n, int W, int W2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= proj_n) return;
    int f = idx / W;
    int w = idx % W;
    buf[(size_t)f * W2 + w].x = proj[idx];
    buf[(size_t)f * W2 + w].y = 0.0f;
}

__global__ void apply_ramp_kernel(cufftComplex* buf, int batch, int W2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * W2;
    if (idx >= total) return;
    int w = idx % W2;
    int k = w <= W2 / 2 ? w : w - W2;
    float ramp = fabsf((float)k) / W2;
    buf[idx].x *= ramp;
    buf[idx].y *= ramp;
}

__global__ void unpad_kernel(const cufftComplex* buf, float* proj, int proj_n, int W, int W2, float inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= proj_n) return;
    int f = idx / W;
    int w = idx % W;
    proj[idx] = buf[(size_t)f * W2 + w].x * inv;
}

int main(int argc, char** argv) {
    if (argc < 15) {
        fprintf(stderr, "Usage: fdk.exe <proj.bin> <vol.bin> <F> <H> <W> <N> <R> <D> <det_pitch> <u0> <v0> <vol_pitch> <span_deg> <filter>\n");
        return 1;
    }
    const char* proj_path = argv[1];
    const char* vol_path = argv[2];
    int F = atoi(argv[3]), H = atoi(argv[4]), W = atoi(argv[5]), N = atoi(argv[6]);
    float R = (float)atof(argv[7]), D = (float)atof(argv[8]);
    float det_pitch = (float)atof(argv[9]), u0 = (float)atof(argv[10]), v0 = (float)atof(argv[11]);
    float vol_pitch = (float)atof(argv[12]);
    float span_deg = (float)atof(argv[13]);
    int filter = atoi(argv[14]);

    size_t proj_n = (size_t)F * H * W;
    size_t vol_n = (size_t)N * N * N;
    printf("[fdk] F=%d H=%d W=%d N=%d R=%.0f D=%.0f span=%.1f filter=%d\n", F, H, W, N, R, D, span_deg, filter);

    // read projections
    float* h_proj = (float*)malloc(proj_n * sizeof(float));
    FILE* fp = fopen(proj_path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", proj_path); return 1; }
    fread(h_proj, sizeof(float), proj_n, fp);
    fclose(fp);

    // angles
    float* cosb = (float*)malloc(F * sizeof(float));
    float* sinb = (float*)malloc(F * sizeof(float));
    for (int f = 0; f < F; f++) {
        float beta = (float)(span_deg * f / F) * (float)M_PI / 180.0f;
        cosb[f] = cosf(beta); sinb[f] = sinf(beta);
    }

    // device allocation
    float *d_proj, *d_vol, *d_cos, *d_sin;
    CUDA_CHECK(cudaMalloc(&d_proj, proj_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vol, vol_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, F * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj, h_proj, proj_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, cosb, F * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sinb, F * sizeof(float), cudaMemcpyHostToDevice));

    // 1. weighting
    {
        int threads = 256;
        int blocks = (int)((proj_n + threads - 1) / threads);
        weight_kernel<<<blocks, threads>>>(d_proj, d_proj, (int)proj_n, H, W, D, det_pitch, u0, v0);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. ramp filter (optional, CUFFT along W)
    if (filter) {
        int batch = F * H, W2 = 2 * W;
        cufftComplex* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, (size_t)batch * W2 * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMemset(d_buf, 0, (size_t)batch * W2 * sizeof(cufftComplex)));
        { int threads = 256, blocks = (int)((proj_n + threads - 1) / threads);
          pad_kernel<<<blocks, threads>>>(d_proj, d_buf, (int)proj_n, W, W2); CUDA_CHECK(cudaGetLastError()); }
        cufftHandle pf, pi;
        CUFFT_CHECK(cufftPlanMany(&pf, 1, &W2, NULL, 1, W2, NULL, 1, W2, CUFFT_C2C, batch));
        CUFFT_CHECK(cufftPlanMany(&pi, 1, &W2, NULL, 1, W2, NULL, 1, W2, CUFFT_C2C, batch));
        CUFFT_CHECK(cufftExecC2C(pf, d_buf, d_buf, CUFFT_FORWARD));
        { int total = batch * W2, threads = 256, blocks = (total + threads - 1) / threads;
          apply_ramp_kernel<<<blocks, threads>>>(d_buf, batch, W2); CUDA_CHECK(cudaGetLastError()); }
        CUFFT_CHECK(cufftExecC2C(pi, d_buf, d_buf, CUFFT_INVERSE));
        { int threads = 256, blocks = (int)((proj_n + threads - 1) / threads);
          unpad_kernel<<<blocks, threads>>>(d_buf, d_proj, (int)proj_n, W, W2, 1.0f / W2); CUDA_CHECK(cudaGetLastError()); }
        CUFFT_CHECK(cufftDestroy(pf)); CUFFT_CHECK(cufftDestroy(pi));
        CUDA_CHECK(cudaFree(d_buf));
    }

    // 3. backprojection
    {
        dim3 block(8, 8, 8);
        dim3 grid((N + 7) / 8, (N + 7) / 8, (N + 7) / 8);
        backproject_kernel<<<grid, block>>>(d_proj, d_vol, F, H, W, N, d_cos, d_sin, R, D, det_pitch, u0, v0, vol_pitch);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // write back
    float* h_vol = (float*)malloc(vol_n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_vol, d_vol, vol_n * sizeof(float), cudaMemcpyDeviceToHost));
    fp = fopen(vol_path, "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", vol_path); return 1; }
    fwrite(h_vol, sizeof(float), vol_n, fp);
    fclose(fp);
    printf("[fdk] wrote %s\n", vol_path);

    CUDA_CHECK(cudaFree(d_proj)); CUDA_CHECK(cudaFree(d_vol));
    CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));
    free(h_proj); free(cosb); free(sinb); free(h_vol);
    return 0;
}
