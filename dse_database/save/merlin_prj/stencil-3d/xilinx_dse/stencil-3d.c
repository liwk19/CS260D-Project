//#include "merlin_type_define.h"
#pragma ACCEL kernel

void stencil3d(long C0,long C1,long orig[39304],long sol[32768])
{
  long sum0;
  long sum1;
  long mul0;
  long mul1;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (long i = (long )1; i < ((long )(34 - 1)); i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (long j = (long )1; j < ((long )(34 - 1)); j++) {
/* Standardize from: for(long ko =(long )1;ko <((long )(34 - 1));ko +=1) {...} */
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (long _in_ko = 1; _in_ko <= 32L; _in_ko++) {
        //long _in_ko = 1L + 1L * ko;
// #define UNROLL_IMPL(k)                                         \
//           sum0 = orig[INDX(row_size, col_size, k, j, i)];      \
//           sum1 = orig[INDX(row_size, col_size, k, j, i + 1)] + \
//                  orig[INDX(row_size, col_size, k, j, i - 1)] + \
//                  orig[INDX(row_size, col_size, k, j + 1, i)] + \
//                  orig[INDX(row_size, col_size, k, j - 1, i)] + \
//                  orig[INDX(row_size, col_size, k + 1, j, i)] + \
//                  orig[INDX(row_size, col_size, k - 1, j, i)];  \
//           mul0 = sum0 * C0;                                    \
//           mul1 = sum1 * C1;                                    \
//           sol[INDX(row_size, col_size, k, j, i)] = mul0 + mul1;
        sum0 = orig[_in_ko + 34 * (j + 34 * i)];
        sum1 = orig[_in_ko + 34 * (j + 34 * (i + 1))] + 
               orig[_in_ko + 34 * (j + 34 * (i - 1))] + 
               orig[_in_ko + 34 * (j + 1 + 34 * i)] + 
               orig[_in_ko + 34 * (j - 1 + 34 * i)] + 
               orig[_in_ko + 1 + 34 * (j + 34 * i)] + 
               orig[_in_ko - 1 + 34 * (j + 34 * i)];
        mul0 = sum0 * C0;
        mul1 = sum1 * C1;
        sol[_in_ko + 34 * (j + 34 * i)] = mul0 + mul1;
      }
    }
  }
}
