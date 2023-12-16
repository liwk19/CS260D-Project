/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "correlation.h"
/* Array initialization. */

static void init_array(int m,int n,double *float_n,double data[100][80])
{
  int i;
  int j;
   *float_n = ((double )100);
  for (i = 0; i < 100; i++) 
    for (j = 0; j < 80; j++) 
      data[i][j] = ((double )(i * j)) / 80 + i;
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

static void print_array(int m,double corr[80][80])
{
  int i;
  int j;
  fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr,"begin dump: %s","corr");
  for (i = 0; i < m; i++) 
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) 
        fprintf(stderr,"\n");
      fprintf(stderr,"%0.2lf ",corr[i][j]);
    }
  fprintf(stderr,"\nend   dump: %s\n","corr");
  fprintf(stderr,"==END   DUMP_ARRAYS==\n");
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void __merlin_dummy_kernel_pragma();

void kernel_correlation(int m,int n,double float_n,double data[100][80],double corr[80][80],double mean[80],double stddev[80])
{
  int i;
  int j;
  int k;
  double eps = 0.1;
  for (j = 0; j < 80; j++) {
    mean[j] = 0.0;
    for (i = 0; i < 100; i++) {
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }
  for (j = 0; j < 80; j++) {
    stddev[j] = 0.0;
    for (i = 0; i < 100; i++) {
      stddev[j] += pow(data[i][j] - mean[j],2);
    }
    stddev[j] /= float_n;
    stddev[j] = sqrt(stddev[j]);
/* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
    stddev[j] = (stddev[j] <= eps?1.0 : stddev[j]);
  }
/* Center and reduce the column vectors. */
  for (i = 0; i < 100; i++) {
    for (j = 0; j < 80; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  }
/* Calculate the m * m correlation matrix. */
  for (i = 0; i < 80 - 1; i++) {
    corr[i][i] = 1.0;
    for (j = i + 1; j < 80; j++) {
      corr[i][j] = 0.0;
      for (k = 0; k < 100; k++) 
        corr[i][j] += data[k][i] * data[k][j];
      corr[j][i] = corr[i][j];
    }
  }
  corr[80 - 1][80 - 1] = 1.0;
}

int main(int argc,char **argv)
{
/* Retrieve problem size. */
  int n = 100;
  int m = 80;
/* Variable declaration/allocation. */
  double float_n;
  double (*data)[100][80];
  data = ((double (*)[100][80])(polybench_alloc_data(((100 + 0) * (80 + 0)),(sizeof(double )))));
  ;
  double (*corr)[80][80];
  corr = ((double (*)[80][80])(polybench_alloc_data(((80 + 0) * (80 + 0)),(sizeof(double )))));
  ;
  double (*mean)[80];
  mean = ((double (*)[80])(polybench_alloc_data((80 + 0),(sizeof(double )))));
  ;
  double (*stddev)[80];
  stddev = ((double (*)[80])(polybench_alloc_data((80 + 0),(sizeof(double )))));
  ;
/* Initialize array(s). */
  init_array(m,n,&float_n, *data);
/* Start timer. */
  ;
/* Run kernel. */
  kernel_correlation(m,n,float_n, *data, *corr, *mean, *stddev);
/* Stop and print timer. */
  ;
  ;
/* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (argc > 42 && !strcmp(argv[0],"")) 
    print_array(m, *corr);
/* Be clean. */
  free((void *)data);
  ;
  free((void *)corr);
  ;
  free((void *)mean);
  ;
  free((void *)stddev);
  ;
  return 0;
}
