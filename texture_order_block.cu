#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define DATATYPE int
#define SMEMSIZE 512
#define REP 128

texture <int,1,cudaReadModeElementType> texref1;
texture <int,1,cudaReadModeElementType> texref2;
__global__ void texture_order_1(double *time,DATATYPE *out,int its)
{
	DATATYPE p,q=threadIdx.x;
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

int main_test(int blocks,int threads,DATATYPE *h_in1,DATATYPE *h_in2,int xxx)
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

	texture_order_1<<<blocks,threads>>>(d_time,d_out,its);
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
	printf("%d\t%d\t%d\t\t%f\t%f\t%f\n",xxx, blocks,threads,avert,mint,maxt);
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
void init_disordered_32(DATATYPE *a,int n)
{
	DATATYPE p[32];
	for (int i=0;i<32;i++)
	{
		p[i]=i;
	}
	for (int i=0;i<n;i+=32)
	{
		for (int j=0;j<32;j++)
		{
			int jj=rand()%(32-j);
			a[i+j]=p[jj];
			for (int k=jj;k<(32-j);k++)
			{
				p[k]=p[k+1];
			}
		}
		for (int j=0;j<32;j++)
		{
			p[j]=a[i+j];
			a[i+j]+=i;
		}
	}
}
void init_disordered_512(DATATYPE *a,int n)
{
	const int nn=n/32;
	DATATYPE *q=(DATATYPE*)malloc(sizeof(DATATYPE)*nn);
	DATATYPE *b=(DATATYPE*)malloc(sizeof(DATATYPE)*n);
	init_order(q,nn);

	for (int i=0;i<n;i+=nn)
	{
		for (int j=0;j<nn;j++)
		{
			int jj=rand()%(nn-j);
			b[i+j]=q[jj];
			for (int k=jj;k<(nn-j);k++)
			{
				q[k]=q[k+1];
			}
		}
		for (int j=0;j<nn;j++)
		{
			q[j]=b[i+j];
		}
	}

	DATATYPE p[32];
	for (int i=0;i<32;i++)
	{
		p[i]=i;
	}
	for (int i=0;i<32;i++)
	{
		for (int j=0;j<nn;j++)
		{
			a[j*32+i]=b[i*nn+j]*32+p[i];
		}
	}
	free(q);
	free(b);
}

int main()
{
	DATATYPE *h_in1, *h_in2, *h_in3;
	h_in1 = (DATATYPE *) malloc(sizeof(DATATYPE) * SMEMSIZE);
	h_in2 = (DATATYPE *) malloc(sizeof(DATATYPE) * SMEMSIZE);
	h_in3 = (DATATYPE *) malloc(sizeof(DATATYPE) * SMEMSIZE);
	init_order(h_in1, SMEMSIZE);
	init_disordered_32(h_in2, SMEMSIZE);
	init_disordered_512(h_in3, SMEMSIZE);

	printf("blocks\t threads\t aver \t min \t max \t(clocks)\n");

	for (int i = 0; i <= 1024; i += 32) {
		int blocks = (i == 0 ? 1 : i);
		int threads = 256;
		main_test(blocks, threads, h_in1, h_in1, 1);
		main_test(blocks, threads, h_in2, h_in2, 2);
		main_test(blocks, threads, h_in3, h_in3, 3);
	}

	free(h_in1);
	free(h_in2);
	free(h_in3);

	return 0;
}