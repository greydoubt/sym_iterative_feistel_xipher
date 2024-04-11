# sym_iterative_feistel_xipher
Symmetric Iterative Feistel Xipher

encryption.cuh contains the functions for performing the work of encryption and decryption. The amount of work for both permute64 and unpermute64 can be controlled via num_iters.
helpers.cuh contains code for error handling, as well as the handy Timer class used throughout this course.
baseline.cu contains code to encrypt on the CPU using multiple CPU cores, transfer data to the GPU, decrypt on the GPU, transfer back to the CPU, and check for correctness on the CPU.
The Timer class is used throughout to give time durations for various portions of the application.
All though it is not required, make can be used to compile code and generate report files. See Makefile for details.
See comments throughout source code for additional details

