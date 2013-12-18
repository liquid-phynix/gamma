all: main
main.o: main.cpp arrays.hpp fft.hpp problem.hpp technicality.hpp error.hpp npy.h math.hpp
	source ~/bin/gcc-4.7-vars; g++ -std=c++11 -I/usr/local/cuda/include -I. -c main.cpp -o main.o
gpu.o: gpu.cu error.hpp
	nvcc -arch=sm_20 -o gpu.o -c gpu.cu
main: main.o gpu.o
	source ~/bin/gcc-4.7-vars; g++ -L/usr/local/cuda/lib64 -lcudart -lcufft -o main main.o gpu.o libnpy.a

# main: main.cu technicality.hpp problem.hpp init.hpp arrays.hpp fft.hpp Makefile
# 	nvcc -Xcompiler='-std=c++0x' -arch=sm_20 -I. -lcufft -o main main.cu lib
# npy.a
# rem:
# 	(rm *.vti; rm *.pks; rm *.line; rm *.aline)
