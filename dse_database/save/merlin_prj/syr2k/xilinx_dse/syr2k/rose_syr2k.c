/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "syr2k.h"
/* Array initialization. */

static void init_array(int n,int m,double *alpha,double *beta,double C[80][80],double A[80][60],double B[80][60])
{
  int i;
  int j;
   *alpha = 1.5;
   *beta = 1.2;
  for (i = 0; i < n; i++) 
    for (j = 0; j < m; j++) {
      A[i][j] = ((double )((i * j + 1) % n)) / n;
      B[i][j] = ((double )((i * j + 2) % m)) / m;
    }
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) {
      C[i][j] = ((double )((i * j + 3) % n)) / m;
    }
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int n,double C[80][80])
{
  int i;
  int j;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","C");
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) 
        fprintf(stderr,"\n");
      fprintf(stderr,"%0.2lf ",C[i][j]);
    }
  fprintf(stderr,"\nend   dump: %s\n","C");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __merlin_dummy_kernel_pragma();

void kernel_syr2k(int n,int m,double alpha,double beta,double C[80][80],double A[80][60],double B[80][60])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
  for (i = 0; i < 80; i++) {
    for (j = 0; j < 80; j++) {
      if (j <= i) {
        C[i][j] *= beta;
      }
    }
    for (k = 0; k < 60; k++) {
      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
        }
      }
    }
  }
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int n = 80;
  int m = 60;
/* Variable declaration/allocation. */
  double alpha;
  double beta;
  double (*C)[80][80];
  C = ((double (*)[80][80])(polybench_alloc_data(((80 + 0) * (80 + 0)),(sizeof(double )))));
  ;
  double (*A)[80][60];
  A = ((double (*)[80][60])(polybench_alloc_data(((80 + 0) * (60 + 0)),(sizeof(double )))));
  ;
  double (*B)[80][60];
  B = ((double (*)[80][60])(polybench_alloc_data(((80 + 0) * (60 + 0)),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(n,m,&alpha,&beta, *C, *A, *B);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_syr2k(n,m,alpha,beta, *C, *A, *B);
/* Stop and print timer. */
  ;
  ;
/* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (argc > 42 && !strcmp(argv[0],"")) 
    print_array(n, *C);
/* Be clean. */
  free((void *)C);
  ;
  free((void *)A);
  ;
  free((void *)B);
  ;
  return 0;
}
