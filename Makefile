STDCPP=/usr/local/gcc-4.7/lib/gcc/x86_64-unknown-linux-gnu/4.7.0/../../../../lib64/libstdc++.a
all: main
main.o: main.cpp arrays.hpp fft.hpp problem.hpp technicality.hpp error.hpp npyutil.hpp math.hpp type.hpp
	source ~/bin/gcc-4.7-vars; g++ -std=c++11 -I/usr/local/cuda/include -I. -c main.cpp -o main.o
gpu.o: gpu.cu error.hpp type.hpp
	nvcc -arch=sm_13 -o gpu.o -c gpu.cu
main: main.o gpu.o
	source ~/bin/gcc-4.7-vars; g++ -static-libgcc -L/usr/local/cuda/lib64 -lcudart -lcufft -o main main.o gpu.o $(STDCPP)
#main: main.o gpu.o
#	source ~/bin/gcc-4.7-vars; g++ -L/usr/local/cuda/lib64 -lcudart -lcufft -o main main.o gpu.o


# main: main.cu technicality.hpp problem.hpp init.hpp arrays.hpp fft.hpp Makefile
# 	nvcc -Xcompiler='-std=c++0x' -arch=sm_20 -I. -lcufft -o main main.cu lib
# npy.a
# rem:
# 	(rm *.vti; rm *.pks; rm *.line; rm *.aline)
