template <typename T> struct TypeHolder {};
template <> struct TypeHolder <float> { typedef float  THReal; typedef float2 THComplex; };
template <> struct TypeHolder <double> { typedef double  THReal; typedef double2 THComplex; };

template <typename T> class Array {
protected:
  T* m_array;
  int3 m_real_dims, m_complex_dims;
  int m_bytes;
public:
  typedef typename TypeHolder<T>::THReal    RealType;
  typedef typename TypeHolder<T>::THComplex ComplexType;
  Array(int3 dims):
    m_array(0),
    m_real_dims(dims){
    m_complex_dims = { dims.x, dims.y, dims.z / 2 + 1 };
    m_bytes = 2 * sizeof(T) * dims.x * dims.y * (dims.z / 2 + 1); }
  void*        void_ptr()     { return m_array; }
  RealType*    real_ptr()     { return m_array; }
  ComplexType* complex_ptr()  { return reinterpret_cast<ComplexType*>(m_array); }
  int3         real_dims()    { return m_real_dims; }
  int3         complex_dims() { return m_complex_dims; }
  int real_axis(int ax){
    assert(ax >= 0 and ax < 3 and "axis index must be in 0..2");
    return *((int*)&m_real_dims + ax); }
  int complex_axis(int ax){
    assert(ax >= 0 and ax < 3 and "axis index must be in 0..2");
    return *((int*)&m_complex_dims + ax); }
};

template <typename T> class CPUArray;

template <typename T> struct GPUArray : public Array <T> {
  GPUArray(int3 dims): Array<T>(dims){
    CUERR(cudaMalloc((void**)&this->m_array, this->m_bytes)); }
  ~GPUArray(){ CUERR(cudaFree(this->m_array)); }
  void ovwrt_with(GPUArray<T>& arr){
    assert(this->real_dims() == arr.real_dims()
           and "source and target array dimensions don't match up");
    cudaMemcpy(this->m_array, arr.void_ptr(), this->m_bytes, cudaMemcpyDeviceToDevice); }
  void ovwrt_with(CPUArray<T>&);
};

template <typename T> struct CPUArray : public Array <T> {
  CPUArray(int3 dims): Array<T>(dims){
    CUERR(cudaHostAlloc((void**)&this->m_array, this->m_bytes, cudaHostAllocDefault));
    memset(this->void_ptr(), 0, this->m_bytes); }
  ~CPUArray(){ CUERR(cudaFreeHost(this->m_array)); }
  inline T& operator[](int3 idx){ return this->m_array[idx.x * this->m_real_dims.y * this->m_real_dims.z + idx.y * this->m_real_dims.z + idx.z]; }
  void ovwrt_with(GPUArray<T>& arr){
    assert(this->real_dims() == arr.real_dims()
           and "source and target array dimensions dont match up");
    cudaMemcpy(this->m_array, arr.void_ptr(), this->m_bytes , cudaMemcpyDeviceToHost); }
  typedef typename Array<T>::RealType*                RealArrType;
  typedef std::complex<typename Array<T>::RealType> * ComplexArrType;
  void save_as_real(const char* fmt, const int it){
    char fn[256]; sprintf(fn, fmt, it);
    aoba::SaveArrayAsNumpy(fn,
                           this->real_axis(0),
                           this->real_axis(1),
                           this->real_axis(2),
                           this->real_ptr()); }
  void save_as_complex(const char* fmt, const int it){
    char fn[256]; sprintf(fn, fmt, it);
    aoba::SaveArrayAsNumpy(fn,
                           this->complex_axis(0),
                           this->complex_axis(1),
                           this->complex_axis(2),
                           this->complex_ptr()); }
};

template <typename T> void GPUArray<T>::ovwrt_with(CPUArray<T>& arr){
  assert(this->real_dims() == arr.real_dims()
         and "source and target array dimensions dont match up");
  cudaMemcpy(this->m_array, arr.void_ptr(), this->m_bytes, cudaMemcpyHostToDevice); }
