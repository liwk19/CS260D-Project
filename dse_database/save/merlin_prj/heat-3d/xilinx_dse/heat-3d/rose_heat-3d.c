/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "heat-3d.h"
/* Array initialization. */

static void init_array(int n,double A[20][20][20],double B[20][20][20])
{
  int i;
  int j;
  int k;
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) 
      for (k = 0; k < n; k++) 
        A[i][j][k] = B[i][j][k] = ((double )(i + j + (n - k))) * 10 / n;
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int n,double A[20][20][20])
{
  int i;
  int j;
  int k;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","A");
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) 
      for (k = 0; k < n; k++) {
        if ((i * n * n + j * n + k) % 20 == 0) 
          fprintf(stderr,"\n");
        fprintf(stderr,"%0.2lf ",A[i][j][k]);
      }
  fprintf(stderr,"\nend   dump: %s\n","A");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __merlin_dummy_kernel_pragma();

void kernel_heat_3d(int tsteps,int n,double A[20][20][20],double B[20][20][20])
{
  int t;
  int i;
  int j;
  int k;
//#pragma scop
  for (t = 1; t <= 40; t++) {
    for (i = 1; i < 20 - 1; i++) {
      for (j = 1; j < 20 - 1; j++) {
        for (k = 1; k < 20 - 1; k++) {
          B[i][j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k]) + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k]) + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1]) + A[i][j][k];
        }
      }
    }
    for (i = 1; i < 20 - 1; i++) {
      for (j = 1; j < 20 - 1; j++) {
        for (k = 1; k < 20 - 1; k++) {
          A[i][j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k]) + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k]) + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1]) + B[i][j][k];
        }
      }
    }
  }
//#pragma endscop
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int n = 20;
  int tsteps = 40;
/* Variable declaration/allocation. */
  double (*A)[20][20][20];
  A = ((double (*)[20][20][20])(polybench_alloc_data(((20 + 0) * (20 + 0) * (20 + 0)),(sizeof(double )))));
  ;
  double (*B)[20][20][20];
  B = ((double (*)[20][20][20])(polybench_alloc_data(((20 + 0) * (20 + 0) * (20 + 0)),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(n, *A, *B);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_heat_3d(tsteps,n, *A, *B);
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
