/*
Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER = gnu

Compiler flags:

BSIZE = 64 # to set block size
CC = cc
OPT = -O3 -unroll-loops -mavx2
CFLAGS = -Wall -std=gnu99 $(OPT) -D BLOCK_SIZE=$(BSIZE)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt  -I$(MKLROOT)/include -Wl,-L$(MKLROOT)/lib/intel64/ -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
*/

#define min(a, b) (((a) < (b)) ? (a) : (b))
// get AVX intrinsics
#include <immintrin.h>
char* dgemm_desc = "Gantzler, Henle, Stark";


// pad a copy of a matrix to an even size
static void pad_matrix(double* padded_matrix, double* matrix, int lda, int padded_lda)
{
  for (int j = 0; j < lda; j++)
  {
    for (int i = 0; i < lda; i++)
    {
      padded_matrix[i + j * padded_lda] = matrix[i + j * lda];
    }
  }
}


// do a block of DGeMM
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C)
{
  // special AVX variable type
  __m128d Aik1, Aik2, Bjk1, Bjk2, Bjk3, Bjk4, Cij1, Cij2;
  for(int k = 0; k < K; k += 2)
  {
    for(int j = 0; j < N; j += 2)
    {
      // load 2x2 sub-block from B
      Bjk1 = _mm_load1_pd(B + k + j * lda); // start of block in B
      Bjk2 = _mm_load1_pd(B + k + 1 + j * lda); // start + 1
      Bjk3 = _mm_load1_pd(B + k + (j + 1) * lda); // start + lda
      Bjk4 = _mm_load1_pd(B + k + 1 + (j + 1) * lda); // start + lda + 1
      // do the math for two elements of C
      for(int i = 0; i < M; i += 2)
      {
        // load 1x2 sub-block from A
        Aik1 = _mm_load_pd(A + i + k * lda);
        Aik2 = _mm_load_pd(A + i + (k + 1) * lda);
        // Load 1x2 sub-block from C
        Cij1 = _mm_load_pd(C + i + j * lda);
        Cij2 = _mm_load_pd(C + i + (j + 1) * lda);
        // Reduce (twice) into C
        Cij1 = _mm_add_pd(_mm_add_pd(Cij1, _mm_mul_pd(Aik1, Bjk1)), _mm_mul_pd(Aik2, Bjk2));
        Cij2 = _mm_add_pd(_mm_add_pd(Cij2, _mm_mul_pd(Aik1, Bjk3)), _mm_mul_pd(Aik2, Bjk4));
        _mm_store_pd(C + i + j * lda, Cij1);
        _mm_store_pd(C + i + (j + 1) * lda, Cij2);
      }
    }
  }
}


// DGeMM on square matrices
void square_dgemm(int lda, double* A, double* B, double* C)
{
  // determine padded_lda for making even-size matrices
  int padded_lda = lda;
  if (lda % 2){
    int t = lda % 2;
    padded_lda = lda + 4 - t;
  }
  // allocate aligned memory for padding input matrices
  double* padded_A = (double*)_mm_malloc(padded_lda * padded_lda * sizeof(double), 8);
  double* padded_B = (double*)_mm_malloc(padded_lda * padded_lda * sizeof(double), 8);
  double* padded_C = (double*)_mm_malloc(padded_lda * padded_lda * sizeof(double), 8);
  // make padded copies of matrices
  pad_matrix(padded_B, B, lda, padded_lda);
  pad_matrix(padded_A, A, lda, padded_lda);
  pad_matrix(padded_C, C, lda, padded_lda);
  /* For each block-row of A */
  for (int i = 0; i < padded_lda; i += BLOCK_SIZE)
  {
    int M = min(BLOCK_SIZE, padded_lda - i);
    /* For each block-column of B */
    for (int j = 0; j < padded_lda; j += BLOCK_SIZE)
    {
      int N = min(BLOCK_SIZE, padded_lda - j);
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < padded_lda; k += BLOCK_SIZE)
      {
         int K = min(BLOCK_SIZE, padded_lda - k);
        /* Perform individual block dgemm */
        do_block(padded_lda, M, N, K, padded_A + i + k * padded_lda,
          padded_B + k + j * padded_lda, padded_C + i + j * padded_lda);
      }
    }
  }
  // un-do padding of C
  for (int j = 0; j < lda; j++)
  {
    for (int i = 0; i < lda; i++)
    {
      C[i + j * lda] = padded_C[i + j * padded_lda];
    }
  }
}
