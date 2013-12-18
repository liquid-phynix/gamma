struct Int3Arg {
  int3 m_int3;
  Int3Arg(): m_int3(){}
  Int3Arg(int a, int b, int c): m_int3({a, b, c}){}
  Int3Arg& operator=(const std::string& str){
    std::istringstream iss(str); char x1, x2;
    bool ret = iss >> m_int3.x >> x1 >> m_int3.y >> x2 >> m_int3.z;
    if(not ret or not (x1 == 'x' || x1 == 'X') or not (x2 == 'x' || x2 == 'X'))
      throw TCLAP::ArgParseException(str + " is not of form %dx%dx%d");
    return *this; }
};
namespace TCLAP { template<> struct ArgTraits<Int3Arg> { typedef StringLike ValueCategory; }; }

std::ostream& operator<<(std::ostream& o, const double3& v){ o << v.x << ' ' << v.y << ' ' << v.z; return o; }
std::ostream& operator<<(std::ostream& o, const int3& v){ o << v.x << ' ' << v.y << ' ' << v.z; return o; }

bool operator==(const int3& a, const int3& b){ return a.x == b.x and a.y == b.y and a.z == b.z; }
//bool operator==(const uint3& a, const uint3& b){ return a.x == b.x and a.y == b.y and a.z == b.z; }
