#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define DATATYPE int
#define SMEMSIZE 2048
#define REP 128
//#define conflictnum 32

texture <int,1,cudaReadModeElementType> texref1;
texture <int,1,cudaReadModeElementType> texref2;
__global__ void texture_broadcast(double *time,DATATYPE *out,int its,int conflictnum)
{
	DATATYPE p,q=(threadIdx.x/conflictnum*conflictnum);
	double time_tmp=0.0;
	unsigned int start_time=0,stop_time=0;
	unsigned int i,j;
	for (i=0;i<its;i++)
	{
		__syncthreads();
		start_time=clock();
#pragma unroll
		for (j=0;j<REP;j++)
		{
			p=tex1Dfetch(texref1,q);
			q=tex1Dfetch(texref2,p);
		}
		stop_time=clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/REP/its;
	out[blockDim.x*blockIdx.x+threadIdx.x] = p+q;
	time[blockDim.x*blockIdx.x+threadIdx.x] = time_tmp;
}

int main_test(int blocks,int threads,DATATYPE *h_in1,DATATYPE *h_in2, int conflictnum)
{
	int its=30;
	DATATYPE *d_in1,*d_in2;
	cudaMalloc((void**)&d_in1,sizeof(DATATYPE)*SMEMSIZE);
	cudaMalloc((void**)&d_in2,sizeof(DATATYPE)*SMEMSIZE);
	cudaMemcpy(d_in1,h_in1,sizeof(DATATYPE)*SMEMSIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2,h_in2,sizeof(DATATYPE)*SMEMSIZE,cudaMemcpyHostToDevice);
	cudaBindTexture(NULL,texref1,d_in1,sizeof(DATATYPE)*SMEMSIZE);
	cudaBindTexture(NULL,texref2,d_in2,sizeof(DATATYPE)*SMEMSIZE);
	double *h_time,*d_time;
	DATATYPE *d_out;
	h_time=(double*)malloc(sizeof(double)*blocks*threads);
	cudaMalloc((void**)&d_time,sizeof(double)*blocks*threads);
	cudaMalloc((void**)&d_out,sizeof(DATATYPE)*blocks*threads);

	texture_broadcast<<<blocks,threads>>>(d_time,d_out,its,conflictnum);
	cudaMemcpy(h_time,d_time,sizeof(double)*blocks*threads,cudaMemcpyDeviceToHost);
	double avert=0.0,maxt=0.0,mint=99999.9;
	int nn=0;
	for (int i=0;i<blocks;i++)
	{
		for (int j=0;j<threads;j+=32)
		{
			avert+=h_time[i*threads+j];
			nn++;
			if (maxt<h_time[i*threads+j])
			{
				maxt=h_time[i*threads+j];
			}
			if (mint>h_time[i*threads+j])
			{
				mint=h_time[i*threads+j];
			}
		}
	}
	avert/=nn;
	printf("%d\t%d\t\t%f\t%f\t%f\n",blocks,threads,avert,mint,maxt);
	cudaUnbindTexture(texref1);
	cudaUnbindTexture(texref2);
	cudaFree(d_time);
	cudaFree(d_out);
	cudaFree(d_in1);
	cudaFree(d_in2);
	free(h_time);
	return 0;
}
void init_order(DATATYPE *a,int n)
{
	for (int i=0;i<n;i++)
	{
		a[i]=i;
	}
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
        printf("%s <conflict> \n", argv[0]);
    } else {

    	int value = atoi(argv[1]);

		DATATYPE *h_in1;
		h_in1 = (DATATYPE*) malloc(sizeof(DATATYPE) * SMEMSIZE);

		init_order(h_in1,SMEMSIZE);

		printf("conflictnum\tblocks\tthreads\taver\tmin\tmax\t(clocks)\n");

		int blocks = 1;
		for (int j = 0; j <= 512; j += 32) {
			int threads = (j == 0 ? 1 : j);
			main_test(blocks, threads, h_in1, h_in1, value);
		}

		free(h_in1);
	}
	return 0;
}