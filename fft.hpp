struct R2C {};
struct C2R {};

template <typename T, typename FT> class CufftPlan {
private:
  cufftHandle m_plan;
  int3 m_dims;
  static cufftType m_type;
public:
  CufftPlan(int3 dims): m_dims(dims){ // logical problem dimensions
    CUFFTERR(cufftPlan3d(&m_plan, dims.x, dims.y, dims.z, m_type));
    // native data layout, nincs paddingelve a valos repr. mint fftw-ben
    CUFFTERR(cufftSetCompatibilityMode(m_plan, CUFFT_COMPATIBILITY_NATIVE)); }
  ~CufftPlan(){ CUFFTERR(cufftDestroy(m_plan)); }
  void execute(GPUArray<T>& arr){ execute(arr, arr); }
  void execute(GPUArray<T>& in, GPUArray<T>& out){
    assert(in.real_dims() == m_dims and out.real_dims() == m_dims
           and "array dimensions in plan and execute phase do not match up");
    switch(m_type){
    case CUFFT_R2C: CUFFTERR(cufftExecR2C(m_plan, in.real_ptr(), out.complex_ptr())); break;
    case CUFFT_C2R: CUFFTERR(cufftExecC2R(m_plan, in.complex_ptr(), out.real_ptr())); break;
    case CUFFT_D2Z: CUFFTERR(cufftExecD2Z(m_plan, in.real_ptr(), out.complex_ptr())); break;
    case CUFFT_Z2D: CUFFTERR(cufftExecZ2D(m_plan, in.complex_ptr(), out.real_ptr())); break;
    default: std::cerr << "this cant be..." << std::endl; }
  CUERR(cudaThreadSynchronize()); }
};

template<> cufftType CufftPlan<float, R2C>::m_type = CUFFT_R2C;
template<> cufftType CufftPlan<float, C2R>::m_type = CUFFT_C2R;
template<> cufftType CufftPlan<double, R2C>::m_type = CUFFT_D2Z;
template<> cufftType CufftPlan<double, C2R>::m_type = CUFFT_Z2D;
