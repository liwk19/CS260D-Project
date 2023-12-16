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
  for (int i = 1; i < 32 - 1; i++) {
    for (int j = 1; j < 32 - 1; j++) {
//    #pragma ss stream nonblock
//  #pragma ss dfg dedicated unroll(1)
      for (int k = 1; k < 16 - 1; k += 2) {
//#define UNROLL_IMPL(k)                                         \

        sum0 = orig[k + 16 * (j + 32 * i)];
        sum1 = orig[k + 16 * (j + 32 * (i + 1))] + orig[k + 16 * (j + 32 * (i - 1))] + orig[k + 16 * (j + 1 + 32 * i)] + orig[k + 16 * (j - 1 + 32 * i)] + orig[k + 1 + 16 * (j + 32 * i)] + orig[k - 1 + 16 * (j + 32 * i)];
        mul0 = sum0 * C0;
        mul1 = sum1 * C1;
        sol[k + 16 * (j + 32 * i)] = mul0 + mul1;
// UNROLL_IMPL(ko + 0)
// UNROLL_IMPL(ko + 1)
      }
    }
  }
//}
}
