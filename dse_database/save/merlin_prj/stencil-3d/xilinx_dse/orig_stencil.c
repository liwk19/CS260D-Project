/*
Implementation based on algorithm described in:
"Stencil computation optimization and auto-tuning on state-of-the-art multicore architectures"
K. Datta, M. Murphy, V. Volkov, S. Williams, J. Carter, L. Oliker, D. Patterson, J. Shalf, K. Yelick
SC 2008
*/

// #include "../Common/test.h"
// #include "../Common/timing.h"

//Define input sizes
#define height_size 32
#define col_size 32
#define row_size 16
//Data Bounds
#define TYPE int
#define MAX 1000
#define MIN 1
//Convenience Macros
#define SIZE (row_size * col_size * height_size)
#define INDX(_row_size,_col_size,_i,_j,_k) ((_i)+_row_size*((_j)+_col_size*(_k)))

#ifndef U
#define U -1
#endif

#pragma ACCEL kernel
void stencil3d(int C0, int C1, int orig[SIZE], int sol[SIZE]) {

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
    int sum0, sum1, mul0, mul1;
    for(int i = 1; i < height_size - 1; i++) {
      for(int j = 1; j < col_size - 1; j++) {
    //    #pragma ss stream nonblock
      //  #pragma ss dfg dedicated unroll(1)
        for(int k = 1; k < row_size - 1; k += 2) {
//#define UNROLL_IMPL(k)                                         \

          sum0 = orig[INDX(row_size, col_size, k, j, i)];      
          sum1 = orig[INDX(row_size, col_size, k, j, i + 1)] + 
                 orig[INDX(row_size, col_size, k, j, i - 1)] + 
                 orig[INDX(row_size, col_size, k, j + 1, i)] + 
                 orig[INDX(row_size, col_size, k, j - 1, i)] + 
                 orig[INDX(row_size, col_size, k + 1, j, i)] + 
                 orig[INDX(row_size, col_size, k - 1, j, i)];  
          mul0 = sum0 * C0;                                    
          mul1 = sum1 * C1;                                    
          sol[INDX(row_size, col_size, k, j, i)] = mul0 + mul1;
          // UNROLL_IMPL(ko + 0)
          // UNROLL_IMPL(ko + 1)
        }
      }
    }
  //}

}

int a[SIZE], b[SIZE], c[2];
int aa[SIZE], bb[SIZE], cc[2];

// int main() {
//   // stencil3d(cc[0], cc[1], aa, bb);
//   stencil3d(c[0], c[1], a, b);
//   begin_roi();
//   stencil3d(c[0], c[1], a, b);
//   end_roi();
//   sb_stats();
// }
