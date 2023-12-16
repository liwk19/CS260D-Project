const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;
#define max(X,Y) ((X)>(Y)?(X):(Y))
#pragma ACCEL kernel

void CnnKernel(const float input[kNum][kInImSize][kInImSize],const float weight[kNum][kNum][kKernel][kKernel],const float bias[kNum],float output[kNum][kOutImSize][kOutImSize])
{
#pragma ACCEL interface variable=input bus_bitwidth=128
  
#pragma ACCEL interface variable=weight bus_bitwidth=128
  
#pragma ACCEL interface variable=output bus_bitwidth=512
  
#pragma ACCEL interface variable=bias bus_bitwidth=512

  float C[kImSize][kImSize];
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < kNum; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (int h = 0; h < kImSize; ++h) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
      for (int w = 0; w < kImSize; ++w) {
        C[h][w] = bias[i];
      }
    }
// Convolution
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (int j = 0; j < kNum; ++j) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L6}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L6}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
      for (int h = 0; h < kImSize; ++h) {
        
#pragma ACCEL PIPELINE auto{__PIPE__L9}
        
#pragma ACCEL TILE FACTOR=auto{__TILE__L9}
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L9}
        for (int w = 0; w < kImSize; ++w) {
          
#pragma ACCEL PIPELINE auto{__PIPE__L10}
          
#pragma ACCEL TILE FACTOR=auto{__TILE__L10}
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
              C[h][w] += weight[i][j][p][q] * input[j][h + p][w + q];
            }
          }
        }
      }
    }
// ReLU
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (int h = 0; h < kImSize; ++h) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
      for (int w = 0; w < kImSize; ++w) {
        C[h][w] = (0.f > C[h][w]?0.f : C[h][w]);
      }
    }
// Max pooling
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (int h = 0; h < kOutImSize; ++h) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L8}
      for (int w = 0; w < kOutImSize; ++w) {
        output[i][h][w] = (((C[h * 2][w * 2] > C[h * 2 + 1][w * 2]?C[h * 2][w * 2] : C[h * 2 + 1][w * 2])) > ((C[h * 2][w * 2 + 1] > C[h * 2 + 1][w * 2 + 1]?C[h * 2][w * 2 + 1] : C[h * 2 + 1][w * 2 + 1]))?((C[h * 2][w * 2] > C[h * 2 + 1][w * 2]?C[h * 2][w * 2] : C[h * 2 + 1][w * 2])) : ((C[h * 2][w * 2 + 1] > C[h * 2 + 1][w * 2 + 1]?C[h * 2][w * 2 + 1] : C[h * 2 + 1][w * 2 + 1])));
      }
    }
  }
}
