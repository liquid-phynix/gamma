#include <cuda.h>
#include <cufft.h>
#include "error.hpp"

#include "type.hpp"

template <typename T1, typename T2> __device__ T1 con(const T2 val){ return val; }

__constant__ Float gpu_eps, gpu_psi0, gpu_mu0, gpu_tau;
__constant__ Float gpu_pv2, gpu_qv2, gpu_pvqv, gpu_denom;
__constant__ Float gpu_pi, gpu_2pi;

__device__ Float2 Dop(Float kp, Float kq, Float kz){
  Float2 ret;
  ret.x = (- gpu_qv2 * kp * kp + con<Float>(2.0) * gpu_pvqv * kp * kq - gpu_pv2 * kq * kq) / gpu_denom - kz * kz;
  ret.y = ret.x * ret.x;
  return ret; }

// __device__ float K(int i, int n, float len){
//   return (i < n / 2 + 1 ? i : i - n) * gpu_2pi / len; }

__device__ Float K(int i, int n){
  return (i < n / 2 + 1 ? i : i - n) * gpu_2pi; }

__global__ void norm_kernel(Float* arr, int len, Float nf){
  for(int i = threadIdx.x; i < len; i += blockDim.x) arr[i] /= nf; }

void norm_kernel_call(Float* arr, int len){
  norm_kernel<<<1, 256>>>(arr, len, len);
  CUERR(cudaPeekAtLastError());
  CUERR(cudaDeviceSynchronize()); }

__global__ void nonlin_kernel(Float* arr, int len){
  for(int i = threadIdx.x; i < len; i += blockDim.x){
    Float tmp = arr[i];
    arr[i] = tmp * tmp * tmp - gpu_mu0; }}

void nonlin_kernel_call(Float* arr, int len){
  nonlin_kernel<<<1, 256>>>(arr, len);
  CUERR(cudaPeekAtLastError());
  CUERR(cudaDeviceSynchronize()); }

// update schemes
// L,L^2 implicit - plain
__device__ Float2 scheme_almost_full(Float2 kpsi, Float2 dop, Float2 knonlin){
  Float denom = con<Float>(1.0) + gpu_tau * (dop.y + con<Float>(2.0) * dop.x);
  Float2 ret = {
    (kpsi.x - gpu_tau * ((con<Float>(1.0) - gpu_eps) * kpsi.x + knonlin.x)) / denom,
    (kpsi.y - gpu_tau * ((con<Float>(1.0) - gpu_eps) * kpsi.y + knonlin.y)) / denom };
  return ret;
}

// __device__ Float2 scheme_gyula(Float2 kpsi, Float2 dop, Float2 knonlin){
//   Float2 ret = {
//     kpsi.x - gpu_tau / (con<Float>(1.0) + gpu_tau * dop.y) * ((con<Float>(1.0) - gpu_eps + con<Float>(2.0) * dop.x + dop.y) * kpsi.x + knonlin.x),
//     kpsi.y - gpu_tau / (con<Float>(1.0) + gpu_tau * dop.y) * ((con<Float>(1.0) - gpu_eps + con<Float>(2.0) * dop.x + dop.y) * kpsi.y + knonlin.y) };
//   return ret;
// }

__global__ void update_kernel(Float2* arr_kpsi, Float2* arr_knonlin, int3 dims, Float lz){
  Float2 kpsi, knonlin;
  Float kzpref = gpu_2pi / lz;
  for(int i = threadIdx.x; i < dims.x; i += blockDim.x){
    for(int j = threadIdx.y; j < dims.y; j += blockDim.y){
      for(int k = threadIdx.z; k < dims.z; k += blockDim.z){
	int idx = i * dims.y * dims.z + j * dims.z + k;
	kpsi = arr_kpsi[idx];
	knonlin = arr_knonlin[idx];
        // scheme selection
        //        arr_kpsi[idx] = scheme_gyula(kpsi, Dop(K(i, dims.x), K(j, dims.y), kzpref * k), knonlin); }}}}
        arr_kpsi[idx] = scheme_almost_full(kpsi, Dop(K(i, dims.x), K(j, dims.y), kzpref * k), knonlin); }}}}


void update_kernel_call(Float2* arr_kpsi, Float2* arr_knonlin, int3 dims, Float lz){
  update_kernel<<<1, dim3(4, 4, 4)>>>(arr_kpsi, arr_knonlin, dims, lz);
  CUERR(cudaPeekAtLastError());
  CUERR(cudaDeviceSynchronize()); }

__global__ void update_kernel_status(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_kenergy, Float2* arr_cpu, int3 dims, Float lz){
  Float2 kpsi, knonlin;
  Float kzpref = gpu_2pi / lz;
  Float2 ret, dop, conv, energy;
  for(int i = threadIdx.x; i < dims.x; i += blockDim.x){
    for(int j = threadIdx.y; j < dims.y; j += blockDim.y){
      for(int k = threadIdx.z; k < dims.z; k += blockDim.z){
	int idx = i * dims.y * dims.z + j * dims.z + k;
	kpsi = arr_kpsi[idx];
	knonlin = arr_knonlin[idx];
	dop = Dop(K(i, dims.x), K(j, dims.y), kzpref * k);
        // scheme selection
        //        ret = scheme_gyula(kpsi, dop, knonlin);
        ret = scheme_almost_full(kpsi, dop, knonlin);
        conv.x = ret.x - kpsi.x;
        conv.y = ret.y - kpsi.y;
        energy.x = (con<Float>(1.0) - gpu_eps + con<Float>(2.0) * dop.x + dop.y) * ret.x;
        energy.y = (con<Float>(1.0) - gpu_eps + con<Float>(2.0) * dop.x + dop.y) * ret.y;
        arr_kenergy[idx] = energy;
        arr_cpu[idx] = conv;
	arr_kpsi[idx] = ret; }}}}

void update_kernel_call_status(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_kenergy, Float2* arr_cpu, int3 dims, Float lz){
  update_kernel_status<<<1, dim3(4, 4, 4)>>>(arr_kpsi, arr_knonlin, arr_kenergy, arr_cpu, dims, lz);
  CUERR(cudaPeekAtLastError());
  CUERR(cudaDeviceSynchronize()); }

/*
// L^2 implicit - gyula - pfcelv
__device__ Float2 scheme_gyula(Float2 kpsi, Float2 dop, Float2 knonlin){
  Float2 ret = {
    kpsi.x - gpu_tau / (1.0f + gpu_tau * dop.y) * ((1.0f - gpu_eps + 2.0f * dop.x + dop.y) * kpsi.x + knonlin.x),
    kpsi.y - gpu_tau / (1.0f + gpu_tau * dop.y) * ((1.0f - gpu_eps + 2.0f * dop.x + dop.y) * kpsi.y + knonlin.y) };
  return ret;
}
// L^2 implicit - plain
__device__ Float2 scheme_plain(Float2 kpsi, Float2 dop, Float2 knonlin){
  Float2 ret = {
    (kpsi.x - gpu_tau * ((1.0f - gpu_eps + 2.0f * dop.x) * kpsi.x + knonlin.x)) / (1.0f + gpu_tau * dop.y),
    (kpsi.y - gpu_tau * ((1.0f - gpu_eps + 2.0f * dop.x) * kpsi.y + knonlin.y)) / (1.0f + gpu_tau * dop.y) };
  return ret;
}
// L,L^2 implicit - rewrite
__device__ Float2 scheme_almost_full_rewrite(Float2 kpsi, Float2 dop, Float2 knonlin){
  Float2 ret = {
    kpsi.x - gpu_tau / (1.0f + gpu_tau * (dop.y + 2.0f * dop.x)) * ((1.0f - gpu_eps + 2.0f * dop.x + dop.y) * kpsi.x + knonlin.x) ,
    kpsi.y - gpu_tau / (1.0f + gpu_tau * (dop.y + 2.0f * dop.x)) * ((1.0f - gpu_eps + 2.0f * dop.x + dop.y) * kpsi.y + knonlin.y) };
  return ret;
}
// full implicit - plain
__device__ Float2 scheme_full(Float2 kpsi, Float2 dop, Float2 knonlin){
  Float2 ret = {
    (kpsi.x - gpu_tau * knonlin.x) / (1.0f + gpu_tau * (1.0f - gpu_eps + 2.0f * dop.x + dop.y)),
    (kpsi.y - gpu_tau * knonlin.y) / (1.0f + gpu_tau * (1.0f - gpu_eps + 2.0f * dop.x + dop.y)) };
  return ret;
}	 
*/
