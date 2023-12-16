/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "trmm.h"
/* Array initialization. */

static void init_array(int m,int n,double *alpha,double A[60][60],double B[60][80])
{
  int i;
  int j;
   *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = ((double )((i + j) % m)) / m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = ((double )((n + (i - j)) % n)) / n;
    }
  }
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int m,int n,double B[60][80])
{
  int i;
  int j;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","B");
  for (i = 0; i < m; i++) 
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) 
        fprintf(stderr,"\n");
      fprintf(stderr,"%0.2lf ",B[i][j]);
    }
  fprintf(stderr,"\nend   dump: %s\n","B");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __merlin_dummy_kernel_pragma();

void kernel_trmm(int m,int n,double alpha,double A[60][60],double B[60][80])
{
//BLAS parameters
//SIDE   = 'L'
//UPLO   = 'L'
//TRANSA = 'T'
//DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM
// B is MxN
  for (int i = 0; i < 60; i++) {
    for (int j = 0; j < 80; j++) {
      for (int k = 0; k < 60; k++) {
        if (k > i) {
          B[i][j] += A[k][i] * B[k][j];
        }
      }
      B[i][j] = alpha * B[i][j];
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
  double (*A)[60][60];
  A = ((double (*)[60][60])(polybench_alloc_data(((60 + 0) * (60 + 0)),(sizeof(double )))));
  ;
  double (*B)[60][80];
  B = ((double (*)[60][80])(polybench_alloc_data(((60 + 0) * (80 + 0)),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(m,n,&alpha, *A, *B);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_trmm(m,n,alpha, *A, *B);
/* Stop and print timer. */
  ;
  ;
/* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (argc > 42 && !strcmp(argv[0],"")) 
    print_array(m,n, *B);
/* Be clean. */
  free((void *)A);
  ;
  free((void *)B);
  ;
  return 0;
}
