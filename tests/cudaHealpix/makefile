test.x: test_pix2ang.cu 
	nvcc -dc ../../src/GpuConvolver/cuda/cudaHealpix.cu -I../../src/
	nvcc -dc test_pix2ang.cu -I/home/pafluxa/.local/include/healpix_cxx/ -I../../src/GpuConvolver/cuda/ 
	nvcc test_pix2ang.o cudaHealpix.o -L/home/pafluxa/.local/lib/ -lhealpix_cxx -o test.x
clean: 
	rm *.o *.x
