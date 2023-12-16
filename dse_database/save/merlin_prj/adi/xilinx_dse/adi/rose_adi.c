/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "adi.h"
/* Array initialization. */

static void init_array(int n,double u[60][60])
{
  int i;
  int j;
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) {
      u[i][j] = ((double )(i + n - j)) / n;
    }
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int n,double u[60][60])
{
  int i;
  int j;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","u");
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) 
        fprintf(stderr,"\n");
      fprintf(stderr,"%0.2lf ",u[i][j]);
    }
  fprintf(stderr,"\nend   dump: %s\n","u");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
//static
static void __merlin_dummy_kernel_pragma();

void kernel_adi(int tsteps,int n,double u[60][60],double v[60][60],double p[60][60],double q[60][60])
{
  int t;
  int i;
  int j;
  double DX;
  double DY;
  double DT;
  double B1;
  double B2;
  double mul1;
  double mul2;
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
//#pragma scop
  DX = 1.0 / ((double )60);
  DY = 1.0 / ((double )60);
  DT = 1.0 / ((double )40);
  B1 = 2.0;
  B2 = 1.0;
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);
  a = -mul1 / 2.0;
  b = 1.0 + mul1;
  c = a;
  d = -mul2 / 2.0;
  e = 1.0 + mul2;
  f = d;
  for (t = 1; t <= 40; t++) {
//Column Sweep
    for (i = 1; i < 60 - 1; i++) {
      v[0][i] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = v[0][i];
      for (j = 1; j < 60 - 1; j++) {
        p[i][j] = -c / (a * p[i][j - 1] + b);
        q[i][j] = (-d * u[j][i - 1] + (1.0 + 2.0 * d) * u[j][i] - f * u[j][i + 1] - a * q[i][j - 1]) / (a * p[i][j - 1] + b);
      }
      v[60 - 1][i] = 1.0;
      for (j = 60 - 2; j >= 1; j--) {
        v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
      }
    }
//Row Sweep
    for (i = 1; i < 60 - 1; i++) {
      u[i][0] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = u[i][0];
      for (j = 1; j < 60 - 1; j++) {
        p[i][j] = -f / (d * p[i][j - 1] + e);
        q[i][j] = (-a * v[i - 1][j] + (1.0 + 2.0 * a) * v[i][j] - c * v[i + 1][j] - d * q[i][j - 1]) / (d * p[i][j - 1] + e);
      }
      u[i][60 - 1] = 1.0;
      for (j = 60 - 2; j >= 1; j--) {
        u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
      }
    }
  }
//#pragma endscop
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int n = 60;
  int tsteps = 40;
/* Variable declaration/allocation. */
  double (*u)[60][60];
  u = ((double (*)[60][60])(polybench_alloc_data(((60 + 0) * (60 + 0)),(sizeof(double )))));
  ;
  double (*v)[60][60];
  v = ((double (*)[60][60])(polybench_alloc_data(((60 + 0) * (60 + 0)),(sizeof(double )))));
  ;
  double (*p)[60][60];
  p = ((double (*)[60][60])(polybench_alloc_data(((60 + 0) * (60 + 0)),(sizeof(double )))));
  ;
  double (*q)[60][60];
  q = ((double (*)[60][60])(polybench_alloc_data(((60 + 0) * (60 + 0)),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(n, *u);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_adi(tsteps,n, *u, *v, *p, *q);
/* Stop and print timer. */
  ;
  ;
/* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (argc > 42 && !strcmp(argv[0],"")) 
    print_array(n, *u);
/* Be clean. */
  free((void *)u);
  ;
  free((void *)v);
  ;
  free((void *)p);
  ;
  free((void *)q);
  ;
  return 0;
}
