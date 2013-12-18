static const char* cufftGetErrorString(cufftResult error){
  switch(error){
  case CUFFT_SUCCESS:        return "CUFFT_SUCCESS";
  case CUFFT_INVALID_PLAN:   return "CUFFT_INVALID_PLAN";
  case CUFFT_ALLOC_FAILED:   return "CUFFT_ALLOC_FAILED";
  case CUFFT_INVALID_TYPE:   return "CUFFT_INVALID_TYPE";
  case CUFFT_INVALID_VALUE:  return "CUFFT_INVALID_VALUE";
  case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
  case CUFFT_EXEC_FAILED:    return "CUFFT_EXEC_FAILED";
  case CUFFT_SETUP_FAILED:   return "CUFFT_SETUP_FAILED";
  case CUFFT_INVALID_SIZE:   return "CUFFT_INVALID_SIZE";
  case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA"; }
  return "<unknown>"; }

#define CUFFTERR(ans) gpuCufftAssert((ans), __FILE__, __LINE__);
inline void gpuCufftAssert(cufftResult code, const char* file, int line){
  if(code != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFTERR: %s %s:%d\n", cufftGetErrorString(code), file, line);
    exit(code); }}

#define CUERR(ans) gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line){
  if(code != cudaSuccess){
    fprintf(stderr, "CUERR: %s %s:%d\n", cudaGetErrorString(code), file, line);
    exit(code); }}
