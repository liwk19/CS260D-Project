/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "atax.h"
/* Array initialization. */

static void init_array(int m,int n,double A[390][410],double x[410])
{
  int i;
  int j;
  double fn;
  fn = ((double )n);
  for (i = 0; i < n; i++) 
    x[i] = 1 + i / fn;
  for (i = 0; i < m; i++) 
    for (j = 0; j < n; j++) 
      A[i][j] = ((double )((i + j) % n)) / (5 * m);
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int n,double y[410])
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

void kernel_atax(int m,int n,double A[390][410],double x[410],double y[410],double tmp[390])
{
  int i;
  int j;
  for (i = 0; i < 410; i++) 
    y[i] = 0;
  for (i = 0; i < 390; i++) {
    tmp[i] = 0.0;
    for (j = 0; j < 410; j++) {
      tmp[i] += A[i][j] * x[j];
    }
    for (j = 0; j < 410; j++) {
      y[j] += A[i][j] * tmp[i];
    }
  }
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int m = 390;
  int n = 410;
/* Variable declaration/allocation. */
  double (*A)[390][410];
  A = ((double (*)[390][410])(polybench_alloc_data(((390 + 0) * (410 + 0)),(sizeof(double )))));
  ;
  double (*x)[410];
  x = ((double (*)[410])(polybench_alloc_data((410 + 0),(sizeof(double )))));
  ;
  double (*y)[410];
  y = ((double (*)[410])(polybench_alloc_data((410 + 0),(sizeof(double )))));
  ;
  double (*tmp)[390];
  tmp = ((double (*)[390])(polybench_alloc_data((390 + 0),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(m,n, *A, *x);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_atax(m,n, *A, *x, *y, *tmp);
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
  free((void *)x);
  ;
  free((void *)y);
  ;
  free((void *)tmp);
  ;
  return 0;
}
