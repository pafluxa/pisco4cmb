test_sky_to_beam_matching_kernel.o: test_sky_to_beam_matching_kernel.cu
	nvcc -dc test_sky_to_beam_matching_kernel.cu -I/home/pafluxa/.local/include/healpix_cxx/ -I../../src/
cudaConv2.o: ../../src/GpuConvolver/cuda/cudaConv2.cu
	nvcc -dc ../../src/GpuConvolver/cuda/cudaConv2.cu -I../../src
cudaSphtransf.o: ../../src/GpuConvolver/cuda/cudaSphtransf.cu
	nvcc -dc ../../src/GpuConvolver/cuda/cudaSphtransf.cu -I../../src
cudaHealpix.o: ../../src/GpuConvolver/cuda/cudaHealpix.cu
	nvcc -dc ../../src/GpuConvolver/cuda/cudaHealpix.cu -I../../src
sphtrigo.o: ../../src/Sphtrigo/sphtrigo.cpp
	g++ -c ../../src/Sphtrigo/sphtrigo.cpp -I../../src/
test.x: test_sky_to_beam_matching_kernel.o cudaHealpix.o cudaSphtransf.o cudaConv2.o sphtrigo.o
	nvcc test_sky_to_beam_matching_kernel.o cudaHealpix.o cudaSphtransf.o cudaConv2.o sphtrigo.o -L/home/pafluxa/.local/lib/ -lhealpix_cxx -o test.x
clean: 
	rm *.o *.x
