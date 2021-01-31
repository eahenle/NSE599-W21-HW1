/*
Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3 -unroll-loops -mavx2
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include <immintrin.h>

const char* dgemm_desc = "Gantzler, Henle, Stark";

#define min(a,b) (((a)<(b))?(a):(b))

///copy array into the padded matrix
static void pad(double* restrict padA, double* restrict A, const int lda, const int newlda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      padA[i + j*newlda] = A[i + j*lda];
    }
  }
}

//copy array into the unpadded matrix
static void unpad(double* restrict padA, double* restrict A, const int lda, const int newlda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      A[i + j*lda] = padA[i + j*newlda];
    }
  }
}

//copy array into the unpadded matrix
// A is lda * ldb
void transposeA(double* restrict AT, double* restrict A, const int lda, const int ldb)
{
  for (int j = 0; j < ldb; j++) {
    for (int i = 0; i < lda; i++) {
      AT[j + i*ldb] = A[i + j*lda];
    }
  }
}

//copy array into the unpadded matrix
void copyA(double* restrict AT, double* restrict A, const int lda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      AT[i + j*lda] = A[i + j*lda];
    }
  }
}

static void do_block(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  __m128d Aik,Aik_1,Bkj,Bkj_1,Bkj_2,Bkj_3,Cij,Cij_1,d0,d1;

  for (int k = 0; k < K; k += 2 ) {
    for (int j = 0; j < N; j += 2) {
      Bkj = _mm_load1_pd(B+k+j*lda);
      Bkj_1 = _mm_load1_pd(B+k+1+j*lda);
      Bkj_2 = _mm_load1_pd(B+k+(j+1)*lda);
      Bkj_3 = _mm_load1_pd(B+k+1+(j+1)*lda);
      for (int i = 0; i < M; i += 2) {
        Aik = _mm_load_pd(A+i+k*lda);
        Aik_1 = _mm_load_pd(A+i+(k+1)*lda);

        Cij = _mm_load_pd(C+i+j*lda);
        Cij_1 = _mm_load_pd(C+i+(j+1)*lda);

        d0 = _mm_add_pd(Cij, _mm_mul_pd(Aik,Bkj));
        d1 = _mm_add_pd(Cij_1, _mm_mul_pd(Aik,Bkj_2));
        Cij = _mm_add_pd(d0, _mm_mul_pd(Aik_1,Bkj_1));
        Cij_1 = _mm_add_pd(d1, _mm_mul_pd(Aik_1,Bkj_3));
        _mm_store_pd(C+i+j*lda,Cij);
        _mm_store_pd(C+i+(j+1)*lda,Cij_1);
      }
    }
  }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (const int lda, double* restrict A, double* restrict B, double* restrict C)
{

  int newlda = lda;
  int div = 8;
  if (lda % div){
    int t = lda % div;
    newlda = lda + (div-t) + div;
  }

  double* padA = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padA, A, lda, newlda);

  double* padB = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padB, B, lda, newlda);

  double* padC = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padC, C, lda, newlda);

  /* For each block-row of A */
  for (int i = 0; i < newlda; i += BLOCK_SIZE)
  {
    /* For each block-column of B */
    for (int j = 0; j < newlda; j += BLOCK_SIZE)
    {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < newlda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        const int M = min (BLOCK_SIZE, newlda-i);
        const int N = min (BLOCK_SIZE, newlda-j);
        const int K = min (BLOCK_SIZE, newlda-k);

        /* Perform individual block dgemm */
        do_block(newlda, M, N, K, padA + i + k*newlda, padB + k + j*newlda, padC + i + j*newlda);
      }
    }
  }

  unpad(padC, C, lda, newlda);
  _mm_free(padA);
  _mm_free(padB);

}
