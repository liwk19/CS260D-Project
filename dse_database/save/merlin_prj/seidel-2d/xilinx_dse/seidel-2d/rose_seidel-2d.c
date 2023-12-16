/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* seidel-2d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "seidel-2d.h"
/* Array initialization. */

static void init_array(int n,double A[120][120])
{
  int i;
  int j;
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) 
      A[i][j] = (((double )i) * (j + 2) + 2) / n;
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int n,double A[120][120])
{
  int i;
  int j;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","A");
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) 
        fprintf(stderr,"\n");
      fprintf(stderr,"%0.2lf ",A[i][j]);
    }
  fprintf(stderr,"\nend   dump: %s\n","A");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __merlin_dummy_kernel_pragma();

void kernel_seidel_2d(int tsteps,int n,double A[120][120])
{
  int t;
  int i;
  int j;
//#pragma scop
  for (t = 0; t <= 40 - 1; t++) {
    for (i = 1; i <= 120 - 2; i++) {
      for (j = 1; j <= 120 - 2; j++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int n = 120;
  int tsteps = 40;
/* Variable declaration/allocation. */
  double (*A)[120][120];
  A = ((double (*)[120][120])(polybench_alloc_data(((120 + 0) * (120 + 0)),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(n, *A);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_seidel_2d(tsteps,n, *A);
/* Stop and print timer. */
  ;
  ;
/* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (argc > 42 && !strcmp(argv[0],"")) 
    print_array(n, *A);
/* Be clean. */
  free((void *)A);
  ;
  return 0;
}
