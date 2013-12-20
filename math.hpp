#define PI M_PI
#define PIX2 (2.0 * M_PI)

typedef boost::rational<int> Irat;

inline int norm2(int3 v){ return v.x * v.x + v.y * v.y + v.z * v.z; }
inline double norm2(double3 v){ return v.x * v.x + v.y * v.y + v.z * v.z; }
inline double norm(double3 v){ return sqrt(norm2(v)); }
inline double norm(int3 v){ return sqrt(norm2(v)); }
inline double3 normalized(int3 v){ double n = norm(v); return { v.x / n, v.y / n, v.z / n }; }
inline double3 normalized(double3 v){ double n = norm(v); return { v.x / n, v.y / n, v.z / n }; }
inline double inner(double3 v1, double3 v2){ return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline double inner(int3 v1, double3 v2){ return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline int3 cross_int3(int3 pv, int3 qv){
  return { pv.y * qv.z - pv.z * qv.y, 
           pv.z * qv.x - pv.x * qv.z,
           pv.x * qv.y - pv.y * qv.x }; }

int3 canonical_miller(int3 m){
  if(m.x == 0 and m.y == 0 and m.z == 0){
    std::cerr << "Miller-indices cannot be zero all at once" << std::endl; exit(-1); }
  if(m.x < 0) m.x = - m.x; if(m.y < 0) m.y = - m.y; if(m.z < 0) m.z = - m.z;
  int vec[3] = {m.x, m.y, m.z}; std::sort(vec, vec + 3);
  int common = boost::gcd(vec[0], boost::gcd(vec[1], vec[2]));
  return { vec[0] / common, vec[1] / common, vec[2] / common}; }

double z_full_mult(int3 pv, int3 qv, double* _d = NULL){
  int3 zv = cross_int3(pv, qv); int denom = norm2(zv);
  Irat cx(zv.x, denom); Irat cy(zv.y, denom); Irat cz(zv.z, denom);
  int mult = boost::lcm(cx.denominator(), boost::lcm(cy.denominator(), cz.denominator()));
  double d = 1.0 / sqrt(denom);
  if(_d) *_d = d;
  std::cout << "crystal plane spacing(sigma=1): " << d << "\n"
            << "z-periodic multiplier(sigma=1): " << mult << std::endl;
  return mult * d; }

double condition_2x2(double* arr){
  double a = arr[0] * arr[0] + arr[2] * arr[2];
  double bc = arr[0] * arr[1] + arr[2] * arr[3];
  double d = arr[1] * arr[1] + arr[3] * arr[3];
  double eig1 = (a + d - sqrt(a * a + 4.0 * bc * bc - 2.0 * a * d + d * d)) / 2.0;
  double eig2 = (a + d + sqrt(a * a + 4.0 * bc * bc - 2.0 * a * d + d * d)) / 2.0;
  assert(eig1 > 0.0 and "eigenvalue must be positive");
  assert(eig2 > 0.0 and "eigenvalue must be positive");
  return sqrt(eig1 > eig2 ? eig1 : eig2); }

double crystal_condition(int3 pv, int3 qv){
  double normP = norm(pv);
  int3 perp = cross_int3(pv, qv);
  int3 newv = cross_int3(pv, perp);
  double3 nvu = normalized(newv);
  double3 pvu = normalized(pv);
  double mat[4] = { normP, inner(qv, pvu),
                    0,     inner(qv, nvu) };
  return condition_2x2(mat); }

int3 integrify(int i1, int i2, Irat r3){
  return {i1 * r3.denominator(), i2 * r3.denominator(), r3.numerator() * r3.denominator()}; }

double find_basis(int m1, int m2, int m3, int3& pv_out, int3& qv_out){
  int mill2 = m1 * m1 + m2 * m2 + m3 * m3;
  const int w = 10;
  double condition = std::numeric_limits<double>::infinity();
  for(int n1 = -w; n1 <= w; n1++){
    for(int n2 = -w; n2 <= w; n2++){
      for(int n3 = -w; n3 <= w; n3++){
        for(int n4 = -w; n4 <= w; n4++){
          int3 pv = integrify(n1, n2, Irat(- (n1 * m1 + n2 * m2), m3));
          int3 qv = integrify(n3, n4, Irat(- (n3 * m1 + n4 * m2), m3));
          int nnn2 = norm2(cross_int3(pv, qv));
          if(mill2 == nnn2){
            double cond = crystal_condition(pv, qv);
            if(cond < condition){
              condition = cond;
              pv_out = pv;
              qv_out = qv; }}}}}}
  assert(condition != std::numeric_limits<double>::infinity() and "find_basis failed");
  return condition; }
