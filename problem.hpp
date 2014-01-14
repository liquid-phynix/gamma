bool is_power_of(int a, int b){
  if(a == 1) return true;
  else if(a == 0) return false;
  else if(a % b == 0) return is_power_of(a / b, b);
  else return false; }

int choose_divs(int len, int N = 4){
  int bases[] = {2, 3, 5, 7};
  for(int i = 0; i < N; i++) if(is_power_of(len, bases[i])) return len;
  int lens[N];
  for(int i = 0; i < N; i++) lens[i] = pow(bases[i], ceil(log(len) / log(bases[i])));
  std::sort(lens, lens + N);
  return lens[0];
}

template <typename T> class Problem {
  const double m_sigma;
  double m_DP, m_DQ;
  int3 m_pv, m_qv;
  double3 m_scaled_pv, m_scaled_qv, m_zv;
  double m_DZ, m_psiL, m_psiS, m_eps;
public:
  int3 m_dims;
  double m_LENZ;
  Problem(int3 miller, int3 dims, bool dims_set,
          double psi_l, double psi_s,
          double sigma_id, double sigma_fact,
          double eps, int zmul):
    m_sigma(sigma_id * sigma_fact){
    m_psiL = psi_l; m_psiS = psi_s; m_eps = eps;

    double base_cond = find_basis(miller.x, miller.y, miller.z, m_pv, m_qv);
    std::cout << "condition number of pgramm base: " << base_cond << std::endl;

    m_scaled_pv = { m_sigma * m_pv.x, m_sigma * m_pv.y, m_sigma * m_pv.z };
    m_scaled_qv = { m_sigma * m_qv.x, m_sigma * m_qv.y, m_sigma * m_qv.z };

    double d;
    //    m_LENZ = m_sigma * zmul * z_full_mult(m_pv, m_qv, &d);
    z_full_mult(m_pv, m_qv, &d);
    m_LENZ = 80.0 * m_sigma;

    if(dims_set) m_dims = dims;
    else {
      double rho = 16.0 / m_sigma;
      m_dims = { choose_divs(int(ceil(rho * norm(m_scaled_pv)))),
                 choose_divs(int(ceil(rho * norm(m_scaled_qv)))),
                 choose_divs(int(ceil(rho * m_LENZ)), 1) }; }
      //    std::cout << "dims = " << m_dims.x << " x " << m_dims.y << " x " << m_dims.z << std::endl;
    // int3 _m_dims = m_dims;
    // //    int3 _m_dims = { 64, 35, 49 };
    // m_dims = { choose_divs(_m_dims.x),
    //            choose_divs(_m_dims.y),
    //            choose_divs(_m_dims.z) };

    std::cout << "dims = " << m_dims.x << " x " << m_dims.y << " x " << m_dims.z << std::endl;

    m_DP = 1.0 / m_dims.x;
    m_DQ = 1.0 / m_dims.y;
    m_DZ = m_LENZ / m_dims.z;

    m_zv = normalized(cross_int3(m_pv, m_qv));
    std::cout << "pv: <" << m_pv << ">; qv: <" << m_qv << ">; zv: <" << m_zv << ">" << std::endl;

    double pv2  = norm2(m_scaled_pv);
    double qv2  = norm2(m_scaled_qv);
    double pvqv = inner(m_scaled_pv, m_scaled_qv);

    uploadValue("gpu_eps",   eps);
    uploadValue("gpu_psi0",  psi_l);
    uploadValue("gpu_mu0",   (1.0 - eps) * psi_l + psi_l * psi_l * psi_l);
    uploadValue("gpu_tau",   1.0e-1);
    uploadValue("gpu_pv2",   pv2);
    uploadValue("gpu_qv2",   qv2);
    uploadValue("gpu_pvqv",  pvqv);
    uploadValue("gpu_denom", pv2 * qv2 - pvqv * pvqv);
    uploadValue("gpu_pi",    PI);
    uploadValue("gpu_2pi",   PIX2); }

  template <typename TT> void uploadValue(const char* var, TT val){
    T tmp = val;
    CUERR(cudaMemcpyToSymbol(var, &tmp, sizeof(T))); }

  double calc_gamma(CPUArray<T>& arr_oper, CPUArray<T>& arr_psi){
    double f0 = (1.0 - m_eps) / 2.0 * m_psiL * m_psiL + m_psiL * m_psiL * m_psiL * m_psiL / 4.0;
    double mu = (1.0 - m_eps) * m_psiL + m_psiL * m_psiL * m_psiL;
    int len = arr_oper.real_axis(0) * arr_oper.real_axis(1) * arr_oper.real_axis(2);
    T* oper_arr = arr_oper.real_ptr();
    T* psi_arr = arr_psi.real_ptr();
    double result = 0;
    for(int i = 0; i < len; i++){
      double psi = psi_arr[i];
      result += psi * oper_arr[i] / 2.0 + psi * psi * psi * psi / 4.0 - f0 + mu * (m_psiL - psi); }
    return m_LENZ * result / (2.0 * len);
  }

  double sm_sc(double x, double y, double z){
    return (cos(PIX2 * x / m_sigma) +
            cos(PIX2 * y / m_sigma) +
            cos(PIX2 * z / m_sigma)) / 3.0; }
  double sm_bcc(double x, double y, double z){
    return (cos(PIX2 * x / m_sigma) * cos(PIX2 * y / m_sigma) +
            cos(PIX2 * y / m_sigma) * cos(PIX2 * z / m_sigma) +
            cos(PIX2 * z / m_sigma) * cos(PIX2 * x / m_sigma)) / 3.0; }
  
  void init_slab(CPUArray<T>& arr, bool full, std::string fn){
    if(fn.length() != 0){
      
      // FILE* fp = fopen(fn.c_str(), "rb");
      // if(fp == NULL){
      //   std::cerr << "initialization file cannot be opened" << fn << std::endl; 
      //   abort(); }
      // fclose(fp);
      // std::vector<T> tmp; int shape[3];
      // aoba::LoadArrayFromNumpy(fn, shape, tmp);
      // std::cerr << "shape: " << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
      // if(arr.real_axis(0) != shape[0] ||
      //    arr.real_axis(1) != shape[1] ||
      //    arr.real_axis(2) != shape[2]){
      //   std::cerr << "input array dimensions dont agree with calculated" << std::endl; 
      //   abort(); }
      // T* arr_ptr = arr.real_ptr();
      // for(int i = 0; i < shape[0] * shape[1] * shape[2]; i++) arr_ptr[i] = tmp[i];

      arr.from_file(fn);
      std::cout << "domain initialized from file" << std::endl;
    }else{
      int3 idx;
      for(idx.x = 0; idx.x < arr.real_axis(0); idx.x++){
        double p = idx.x * m_DP;
        for(idx.y = 0; idx.y < arr.real_axis(1); idx.y++){
          double q = idx.y * m_DQ;
          for(idx.z = 0; idx.z < arr.real_axis(2); idx.z++){
            double _z = idx.z * m_DZ - 0.5 * m_LENZ;
            double x = p * m_scaled_pv.x + q * m_scaled_qv.x + _z * m_zv.x;
            double y = p * m_scaled_pv.y + q * m_scaled_qv.y + _z * m_zv.y;
            double z = p * m_scaled_pv.z + q * m_scaled_qv.z + _z * m_zv.z;
            double env = 0.25 * (1.0 + tanh((_z + m_LENZ / 4.0) / m_sigma)) * (1.0 + tanh((- _z + m_LENZ / 4.0) / m_sigma));
            arr[idx] = full ? sm_bcc(x, y, z) : m_psiL + env * (m_psiS - m_psiL + sm_bcc(x, y, z)); }}}
      std::cout << "domain initialized" << std::endl; }}
};
