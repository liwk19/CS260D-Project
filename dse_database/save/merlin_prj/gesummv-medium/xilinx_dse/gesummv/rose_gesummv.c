/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "gesummv.h"
/* Array initialization. */

static void init_array(int n,double *alpha,double *beta,double A[250][250],double B[250][250],double x[250])
{
  int i;
  int j;
   *alpha = 1.5;
   *beta = 1.2;
  for (i = 0; i < n; i++) {
    x[i] = ((double )(i % n)) / n;
    for (j = 0; j < n; j++) {
      A[i][j] = ((double )((i * j + 1) % n)) / n;
      B[i][j] = ((double )((i * j + 2) % n)) / n;
    }
  }
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int n,double y[250])
{
  int i;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) 
      fprintf(stderr,"\n");
    fprintf(stderr,"%0.2lf ",y[i]);
  }
  fprintf(stderr,"\nend   dump: %s\n","y");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __merlin_dummy_kernel_pragma();

void kernel_gesummv(int n,double alpha,double beta,double A[250][250],double B[250][250],double tmp[250],double x[250],double y[250])
{
  int i;
  int j;
  for (i = 0; i < 250; i++) {
    tmp[i] = 0.0;
    y[i] = 0.0;
    for (j = 0; j < 250; j++) {
      tmp[i] += A[i][j] * x[j];
      y[i] += B[i][j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int n = 250;
/* Variable declaration/allocation. */
  double alpha;
  double beta;
  double (*A)[250][250];
  A = ((double (*)[250][250])(polybench_alloc_data(((250 + 0) * (250 + 0)),(sizeof(double )))));
  ;
  double (*B)[250][250];
  B = ((double (*)[250][250])(polybench_alloc_data(((250 + 0) * (250 + 0)),(sizeof(double )))));
  ;
  double (*tmp)[250];
  tmp = ((double (*)[250])(polybench_alloc_data((250 + 0),(sizeof(double )))));
  ;
  double (*x)[250];
  x = ((double (*)[250])(polybench_alloc_data((250 + 0),(sizeof(double )))));
  ;
  double (*y)[250];
  y = ((double (*)[250])(polybench_alloc_data((250 + 0),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(n,&alpha,&beta, *A, *B, *x);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_gesummv(n,alpha,beta, *A, *B, *tmp, *x, *y);
/* Stop and print timer. */
  ;
  ;
/* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (argc > 42 && !strcmp(argv[0],"")) 
    print_array(n, *y);
/* Be clean. */
  free((void *)A);
  ;
  free((void *)B);
  ;
  free((void *)tmp);
  ;
  free((void *)x);
  ;
  free((void *)y);
  ;
  return 0;
}
