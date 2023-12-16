#include "merlin_type_define.h"
#pragma ACCEL kernel

void stencil3d(int C0,int C1,int orig[16384],int sol[16384])
{
//#pragma ss config
//{
//  #pragma ss stream
//  for(int j=0; j<col_size; j++)
//    #pragma ss dfg datamove
//    for(int k=0; k<row_size; k++)
//      sol[INDX(row_size, col_size, k, j, 0)] = orig[INDX(row_size, col_size, k, j, 0)];
//}
//#pragma ss config
//{
//  #pragma ss stream
//  for(int j=0; j<col_size; j++)
//    #pragma ss dfg datamove
//    for(int k=0; k<row_size; k++)
//      sol[INDX(row_size, col_size, k, j, height_size-1)] = orig[INDX(row_size, col_size, k, j, height_size-1)];
//}
//#pragma ss config
//{
//  #pragma ss stream
//  for(int i=1; i<height_size-1; i++)
//    #pragma ss dfg datamove
//    for(int k=0; k<row_size; k++)
//      sol[INDX(row_size, col_size, k, 0, i)] = orig[INDX(row_size, col_size, k, 0, i)];
//}
//#pragma ss config
//{
//  #pragma ss stream
//  for(int i=1; i<height_size-1; i++)
//    #pragma ss dfg datamove
//    for(int k=0; k<row_size; k++)
//      sol[INDX(row_size, col_size, k, col_size-1, i)] = orig[INDX(row_size, col_size, k, col_size-1, i)];
//}
//#pragma ss config
//{
//  #pragma ss stream
//  for(int i=1; i<height_size-1; i++)
//    #pragma ss dfg datamove
//    for(int j=1; j<col_size-1; j++)
//      sol[INDX(row_size, col_size, 0, j, i)] = orig[INDX(row_size, col_size, 0, j, i)];
//}
//#pragma ss config
//{
//  #pragma ss stream
//  for(int i=1; i<height_size-1; i++)
//    #pragma ss dfg datamove
//    for(int j=1; j<col_size-1; j++)
//      sol[INDX(row_size, col_size, row_size-1, j, i)] = orig[INDX(row_size, col_size, row_size-1, j, i)];
//}
// Stencil computation
//#pragma ss config
//{
  int sum0;
  int sum1;
  int mul0;
  int mul1;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (int i = 1; i < 32 - 1; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (int j = 1; j < 32 - 1; j++) {
//    #pragma ss stream nonblock
//  #pragma ss dfg dedicated unroll(1)
/* Standardize from: for(int k = 1;k < 16 - 1;k += 2) {...} */
      for (int k = 0; k <= 6; k++) {
        int _in_k = 1 + 2L * k;
//#define UNROLL_IMPL(k)                                         \

        sum0 = orig[_in_k + 16 * (j + 32 * i)];
        sum1 = orig[_in_k + 16 * (j + 32 * (i + 1))] + orig[_in_k + 16 * (j + 32 * (i - 1))] + orig[_in_k + 16 * (j + 1 + 32 * i)] + orig[_in_k + 16 * (j - 1 + 32 * i)] + orig[_in_k + 1 + 16 * (j + 32 * i)] + orig[_in_k - 1 + 16 * (j + 32 * i)];
        mul0 = sum0 * C0;
        mul1 = sum1 * C1;
        sol[_in_k + 16 * (j + 32 * i)] = mul0 + mul1;
// UNROLL_IMPL(ko + 0)
// UNROLL_IMPL(ko + 1)
      }
    }
  }
//}
}
