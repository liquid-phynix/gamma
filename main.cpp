#include <iostream>
#include <cstdio>
#include <algorithm>
#include <functional>
#include <complex>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <tclap/CmdLine.h>
#include <boost/rational.hpp>

#include "npy.h"
#include "error.hpp"
#include "technicality.hpp"
#include "math.hpp"
#include "arrays.hpp"
// #include "problem.hpp"
// #include "fft.hpp"

// prototypes from gpu.cu
void update_kernel_call(cufftComplex*, cufftComplex*, uint3, float);
void update_kernel_call_status(cufftComplex*, cufftComplex*, cufftComplex*, cufftComplex*, uint3, float);
void nonlin_kernel_call(cufftReal*, uint);
void norm_kernel_call(cufftReal*, uint);

// double calc_conv(CPUArray& arrr){
//   double conv = 0;
//   int len = arrr.get_dim_complex(0) * arrr.get_dim_complex(1) * arrr.get_dim_complex(2);
//   std::complex<float>* arr = (std::complex<float>*)arrr.get_array();
//   for(int i = 0; i < len; i++) conv += fabs(arr[i]);
//   return sqrt(conv / len);
// }

// double calc_conv_eq(CPUArray& arrr){
//   double conv = 0;
//   int len = arrr.get_dim_real(0) * arrr.get_dim_real(1) * arrr.get_dim_real(2);
//   float* arr = (float*)arrr.get_array();
//   for(int i = 0; i < len; i++) conv += arr[i] * arr[i];
//   return sqrt(conv / len);
// }

// double calc_conv_eq_max(CPUArray& arrr){
//   int len = arrr.get_dim_real(0) * arrr.get_dim_real(1) * arrr.get_dim_real(2);
//   float* arr = (float*)arrr.get_array();
//   double conv = fabs(arr[0]);
//   for(int i = 0; i < len; i++) conv = conv > fabs(arr[i]) ? conv : fabs(arr[i]);
//   return conv;
// }

int main(int argc, char* argv[]){
  int device = 0;
  int3 dims;
  int3 miller;
  int max_iters;
  try{
    TCLAP::CmdLine cmd("Gamma(Miller-indices)", ' ', "0.1");
    TCLAP::ValueArg<int> deviceArg("g", "gpu", "compute device number", false, 0, "gpu device", cmd);
    TCLAP::ValueArg<int> iterArg("i", "iters", "number of iterations", false, 0, "iterations", cmd);
    TCLAP::ValueArg<Int3Arg> dimsArg("d", "dims", "Array dimensions of the problem", true, Int3Arg(), "N0xN1xN2", cmd);
    TCLAP::ValueArg<Int3Arg> millerArg("m", "miller", "Miller-indices of crystal plane", false, Int3Arg(0, 0, 1), "M1xM2xM3", cmd);
    cmd.parse(argc, argv);
    device = deviceArg.getValue();
    dims = dimsArg.getValue().m_int3;
    max_iters = iterArg.getValue();
    miller = canonical_miller(millerArg.getValue().m_int3); }
  catch (TCLAP::ArgException& e){
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

  CUERR(cudaSetDevice(device));
  std::cout << "device <" << device << "> selected" << std::endl;
  std::cout << "array logical dimensions <" << dims.x << "x" << dims.y << "x" << dims.z << ">" << std::endl;
  std::cout << "Canonical Miller-indices <" << miller << ">" << std::endl;

  // CPUArray arr_master(dims);
  // CPUArray arr_master_2(dims);
  // GPUArray arr_pri(dims);
  // GPUArray arr_sec(dims);
  // GPUArray arr_energy(dims);
  // GPUArray arr_gpu_tmp(dims);

  // CufftPlan r2c(CUFFT_R2C, dims);
  // CufftPlan c2r(CUFFT_C2R, dims);

  // Problem prob(miller, dims, PIX2 * sqrt(2.0), 1.0110354, 10.0);
  // prob.init_slab(arr_master);                                                  // r-psi initialized on cpu
  // //  prob.init_full(arr_master);                                                  // r-psi initialized on cpu
  // arr_master.save_real_array("start.npy", 0);
  // arr_pri.ovwrt_with(arr_master);                                              // r-psi uploaded to arr_pr
  // r2c.execute(arr_pri, arr_sec);                                               // k-psi in arr_sec

  // // problem solution
  // int iters = 0;
  // while(iters++ < max_iters){
  //   nonlin_kernel_call((float*)arr_pri.get_array(), dims.x * dims.y * dims.z); // r-nonlin in arr_pri
  //   r2c.execute(arr_pri);                                                      // k-nonlin in A0
    
  //   if(iters % 100 == 0){
  //     std::cout << "iters:\t" << iters << std::endl;
  //     // testing convergence in spectral space
  //     update_kernel_call_status((cufftComplex*)arr_sec.get_array(),            // k-psi
  //                               (cufftComplex*)arr_pri.get_array(),            // k-nonlin
  //                               (cufftComplex*)arr_energy.get_array(),
  //                               //                          (cufftComplex*)arr_master.get_array(),
  //                               (cufftComplex*)arr_gpu_tmp.get_array(),
  //                               arr_pri.get_complex_dims(), prob.m_LENZ);      // k-psi' in arr_sec
  //     c2r.execute(arr_sec, arr_pri);                                           // r-psi' in arr_pri
  //     norm_kernel_call((float*)arr_pri.get_array(), dims.x * dims.y * dims.z);

  //     c2r.execute(arr_gpu_tmp);
  //     norm_kernel_call((float*)arr_gpu_tmp.get_array(), dims.x * dims.y * dims.z);
  //     // cpu side of testing convergence 
  //     arr_master.ovwrt_with(arr_gpu_tmp);
  //     double conv = calc_conv_eq_max(arr_master);
  //     printf("conv:\t%e\n", conv);
  //     // cpu side of calculating interface energy
  //     c2r.execute(arr_energy);
  //     norm_kernel_call((float*)arr_energy.get_array(), dims.x * dims.y * dims.z);
  //     arr_master.ovwrt_with(arr_pri);
  //     arr_master_2.ovwrt_with(arr_energy);
  //     //                             lin. part     r-psi
  //     double gamma = prob.calc_gamma(arr_master_2, arr_master);
  //     printf("gamma:\t%e\n", gamma);
  //   }else{
  //     update_kernel_call((cufftComplex*)arr_sec.get_array(),                   // k-psi
  //                        (cufftComplex*)arr_pri.get_array(),                   // k-nonlin
  //                        arr_pri.get_complex_dims(), prob.m_LENZ);             // k-psi' in arr_sec
  //     c2r.execute(arr_sec, arr_pri);                                           // r-psi' in arr_pri
  //     norm_kernel_call((float*)arr_pri.get_array(), dims.x * dims.y * dims.z);
  //   }

  // }

  // arr_master.ovwrt_with(arr_pri);
  // arr_master.save_real_array("end.npy", 0);

  return 0;
}
