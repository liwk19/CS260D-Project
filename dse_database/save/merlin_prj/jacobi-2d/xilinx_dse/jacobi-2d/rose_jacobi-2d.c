/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-2d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "jacobi-2d.h"
/* Array initialization. */

static void init_array(int n,double A[90][90],double B[90][90])
{
  int i;
  int j;
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) {
      A[i][j] = (((double )i) * (j + 2) + 2) / n;
      B[i][j] = (((double )i) * (j + 3) + 3) / n;
    }
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int n,double A[90][90])
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

void kernel_jacobi_2d(int tsteps,int n,double A[90][90],double B[90][90])
{
  int t;
  int i;
  int j;
//#pragma scop
  for (t = 0; t < 40; t++) {
    for (i = 1; i < 90 - 1; i++) {
      for (j = 1; j < 90 - 1; j++) {
        B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
      }
    }
    for (i = 1; i < 90 - 1; i++) {
      for (j = 1; j < 90 - 1; j++) {
        A[i][j] = 0.2 * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]);
      }
    }
  }
//#pragma endscop
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int n = 90;
  int tsteps = 40;
/* Variable declaration/allocation. */
  double (*A)[90][90];
  A = ((double (*)[90][90])(polybench_alloc_data(((90 + 0) * (90 + 0)),(sizeof(double )))));
  ;
  double (*B)[90][90];
  B = ((double (*)[90][90])(polybench_alloc_data(((90 + 0) * (90 + 0)),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(n, *A, *B);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_jacobi_2d(tsteps,n, *A, *B);
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
  free((void *)B);
  ;
  return 0;
}
