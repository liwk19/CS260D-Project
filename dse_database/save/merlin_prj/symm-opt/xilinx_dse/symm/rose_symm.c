/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "symm.h"
/* Array initialization. */

static void init_array(int m,int n,double *alpha,double *beta,double C[60][80],double A[60][60],double B[60][80])
{
  int i;
  int j;
   *alpha = 1.5;
   *beta = 1.2;
  for (i = 0; i < m; i++) 
    for (j = 0; j < n; j++) {
      C[i][j] = ((double )((i + j) % 100)) / m;
      B[i][j] = ((double )((n + i - j) % 100)) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <= i; j++) 
      A[i][j] = ((double )((i + j) % 100)) / m;
    for (j = i + 1; j < m; j++) 
//regions of arrays that should not be used
      A[i][j] = (- 999);
  }
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int m,int n,double C[60][80])
{
  int i;
  int j;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","C");
  for (i = 0; i < m; i++) 
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) 
        fprintf(stderr,"\n");
      fprintf(stderr,"%0.2lf ",C[i][j]);
    }
  fprintf(stderr,"\nend   dump: %s\n","C");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __merlin_dummy_kernel_pragma();

void kernel_symm(int m,int n,double alpha,double beta,double C[60][80],double A[60][60],double B[60][80])
{
  int i;
  int j;
  int k;
  double temp2;
//BLAS PARAMS
//SIDE = 'L'
//UPLO = 'L'
// =>  Form  C := alpha*A*B + beta*C
// A is MxM
// B is MxN
// C is MxN
//note that due to Fortran array layout, the code below more closely resembles upper triangular case in BLAS
  for (i = 0; i < 60; i++) {
    for (j = 0; j < 80; j++) {
      temp2 = 0;
      for (k = 0; k < 60; k++) {
        if (k < i) {
          C[k][j] += alpha * B[i][j] * A[i][k];
          temp2 += B[k][j] * A[i][k];
        }
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
  }
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int m = 60;
  int n = 80;
/* Variable declaration/allocation. */
  double alpha;
  double beta;
  double (*C)[60][80];
  C = ((double (*)[60][80])(polybench_alloc_data(((60 + 0) * (80 + 0)),(sizeof(double )))));
  ;
  double (*A)[60][60];
  A = ((double (*)[60][60])(polybench_alloc_data(((60 + 0) * (60 + 0)),(sizeof(double )))));
  ;
  double (*B)[60][80];
  B = ((double (*)[60][80])(polybench_alloc_data(((60 + 0) * (80 + 0)),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(m,n,&alpha,&beta, *C, *A, *B);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_symm(m,n,alpha,beta, *C, *A, *B);
/* Stop and print timer. */
  ;
  ;
/* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (argc > 42 && !strcmp(argv[0],"")) 
    print_array(m,n, *C);
/* Be clean. */
  free((void *)C);
  ;
  free((void *)A);
  ;
  free((void *)B);
  ;
  return 0;
}
