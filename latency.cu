#include <stdlib.h>
#include <stdio.h>
#define DATATYPE int
#define ARRAYLEN 2048
#define REP 128
//#define PRINTNEED
#define TIMETESTEVENT
#include <cuda_runtime.h>
#include "repeat.h"

__global__ void test_register_latency(double *time,DATATYPE *out,int its)
{
	int p=3;
	int q=1;
	int r,x=2,y=5,z=7;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(r=p;p=q;q=x;x=y;y=z;z=r;)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0/its;
	out[0] =r;
	time[0] = time_tmp;
}
__constant__ DATATYPE d_const_array[ARRAYLEN];
__global__ void test_const_latency(double *time,DATATYPE *out,int its)
{
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=d_const_array[p];)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0/its;
	out[1] =p;
	time[1] = time_tmp;
}
__global__ void test_shared_latency(double *time,DATATYPE *out,int its,DATATYPE *array)
{
	__shared__ DATATYPE shared_array[ARRAYLEN];
	int i;
	for (i=0;i<ARRAYLEN;i++)
	{
		shared_array[i]=array[i];
	}
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=shared_array[p];)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0/its;
	out[2] =p;
	time[2] = time_tmp;
}
__global__ void test_local_latency(double *time,DATATYPE *out,int its,DATATYPE *array)
{
	DATATYPE local_array[ARRAYLEN];
	int i;
	for (i=0;i<ARRAYLEN;i++)
	{
		local_array[i]=array[i];
	}
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=local_array[p];)
			stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0/its;
	out[3] =p;
	time[3] = time_tmp;
}
__global__ void test_global_latency(double *time,DATATYPE *out,int its,DATATYPE *array)
{
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=array[p];)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0/its;
	out[4] =p;
	time[4] = time_tmp;
}

texture <int,1,cudaReadModeElementType> texref;
__global__ void test_texture_latency(double *time,DATATYPE *out,int its)
{
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=tex1Dfetch(texref,p);)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0/its;
	out[5] =p;
	time[5] = time_tmp;
}



void call_test_latency(int step,int its,double *h_time)
{
	DATATYPE *h_array;
	h_array=(DATATYPE*)malloc(sizeof(DATATYPE)*ARRAYLEN);
	for (int i=0;i<ARRAYLEN;i++)
	{
		h_array[i]=(i+step)%ARRAYLEN;
	}
	DATATYPE *d_array;
	cudaMalloc((void**)&d_array,sizeof(DATATYPE)*ARRAYLEN);
	cudaMemcpy(d_array,h_array,sizeof(DATATYPE)*ARRAYLEN,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_const_array,h_array,sizeof(DATATYPE)*ARRAYLEN);

	/*texture*/
	cudaBindTexture(NULL,texref,d_array,sizeof(DATATYPE)*ARRAYLEN);

	double *d_time;
	cudaMalloc((void**)&d_time,sizeof(double)*6);
	DATATYPE *d_out,*h_out;
	h_out=(DATATYPE *)malloc(sizeof(DATATYPE)*6);
	cudaMalloc((void**)&d_out,sizeof(DATATYPE)*6);

	test_register_latency	<<<1,1>>>(d_time,d_out,its);
	test_const_latency		<<<1,1>>>(d_time,d_out,its);
	test_shared_latency		<<<1,1>>>(d_time,d_out,its,d_array);
	test_local_latency		<<<1,1>>>(d_time,d_out,its,d_array);
	test_global_latency		<<<1,1>>>(d_time,d_out,its,d_array);
	test_texture_latency	<<<1,1>>>(d_time,d_out,its);

	cudaMemcpy(h_out,d_out,sizeof(DATATYPE)*6,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_time,d_time,sizeof(double)*6,cudaMemcpyDeviceToHost);
	printf("%d:\t%f\t%f\t%f\t%f\t%f\t%f\n",step,h_time[0],h_time[1],h_time[2],h_time[3],h_time[4],h_time[5]);
//	printf("out=%d\t%d\t%d\t%d\t%d\n",h_out[0],h_out[1],h_out[2],h_out[3],h_out[4]);

	cudaUnbindTexture(texref);
	cudaFree(d_array);
	cudaFree(d_time);
	cudaFree(d_out);
	free(h_array);
	free(h_out);
}


int  main()
{
	double *h_time;
	h_time = (double *) malloc(sizeof(double) * 6 * 1024);
	printf("step:register\t constant\t shared\t local\t global\t texture\n");
	
	for (int i = 1; i <= 1024; i++) {
		call_test_latency(i, 30, &h_time[(i - 1) * 6]);
	}

	printf("average:\t");
	for (int i = 0; i < 6; i++) {
		double average = 0.0;
		for (int j = 0; j < 1024; j++) {
			average += h_time[j * 6 + i];
		}
		average /= 1024.0;
		printf("%f\t", average);
	}
	printf("\n");
	return 0;
}