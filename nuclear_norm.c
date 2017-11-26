#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// machine precision
#define EPS (3e-16)

// compute svd by LAPACK
void nuclear_norm_lapack(double* X, double* nuc, int row, int col);

/* DGESVD prototype */
extern void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
                    double* A, const int* lda, double* S, double* u, const int* ldu,
                    double* vt, const int* ldvt, double* work, const int* lwork,
                    int* info);

// compute svd by Golub-Reinsch SVD algorithm, 
// that first transform the matrix A into a bidiagonal matrix B
// and solve SVD of B by QR-iteration
void nuclear_norm(double* X, double* nuc, int row, int col);

// compute Householder vector
void house_vec(double* a, int n, int stride, double* v, double* beta);

// compute Wilkinson shift, that is the eigenvalue of a 2x2 matrix (nearer to C_22])
double wilkinson_shift(const double C[]);

int main(int argc, const char *argv[])
{
    // simple test matrix
    //double A[] = {76, 27, 18, 25, 89, 60, 11, 51, 32, 12, 13, 14};
    double nuc = 0.;
    int m = 100, n = 50;

    double* A = (double*) malloc(sizeof(double) * m * n);
    for (int i = 0; i < m * n; ++i) {
        // random real number between [-1, 1]
        A[i] = ((double)rand() / (double)(RAND_MAX)) * 2. - 1.;
    }

    // test my implementation of SVD (only singular values involved)
    nuclear_norm(A, &nuc, m, n);
    printf("nuclear norm: %.6f \n", nuc);

    // LAPACK call will modify A
    nuclear_norm_lapack(A, &nuc, m, n);
    printf("nuclear norm by LAPACK: %.6f \n", nuc);

    free(A);
    return 0;
}

void nuclear_norm_lapack(double* X, double* nuc, int row, int col) {
    int num_sings = row < col ? row: col; 
    double* sings = (double*) malloc(num_sings * sizeof(double));

    int info = 0;
    int lwork = -1;
    double* work;
    double wkopt;

    // query and allocate the optimal workspace
    dgesvd_("N", "N", &row, &col, X, &row, sings, NULL, &row, NULL, &col, &wkopt, &lwork, &info);

    lwork = (int)wkopt;
    work = (double*)malloc(lwork * sizeof(double));

    // compute svd
    dgesvd_("N", "N", &row, &col, X, &row, sings, NULL, &row, NULL, &col, work, &lwork, &info);

    // check for convergence
    if (info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        free(work);
        free(sings);
        exit(1);
    }

    // compute nuclear norm
    *nuc = 0.;
    for (int i = 0; i < num_sings;  ++i) {
        *nuc += sings[i];
    }

    free(work);
    free(sings);
}

void nuclear_norm(double* X, double* nuc, int row, int col) {
    int num_sings = 0;  // numbers of singular values
    int m, n;

    // computation performed on M, so X is not modified
    double* M = (double*) malloc(row * col * sizeof(double));
    // copy X
    if (row < col) {
        m = col;
        n = row;

        // consider M = X^T
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                M[j * m + i] = X[i * row + j];
            }
        }

    } else {
        // M = X
        memcpy(M, X, row * col * sizeof(double));
        m = row;
        n = col;
    }

    num_sings = n;
    *nuc = 0.;

    //for (int j = 0; j < n; ++j) {
    //    for (int i = 0; i < m; ++i) {
    //        printf("M[%d, %d] = %.6f\n", i, j, M[i + j * m]);
    //    }
    //}

    /*******
     * transform matrix X to a bidiagonal matrix by Householder transformation
     *******/
    double* v = (double*) malloc(m * sizeof(double));   // m >= n
    double* temp = (double*)malloc(m * sizeof(double));
    double beta = 0.;
    for (int i = 0; i < n; ++i) {
        // Householder transformation to column
        house_vec(M + (i * m + i), m - i, 1, v, &beta);

        //for (int j = 0; j < m - i; ++j) printf("v[%d] = %.8f\n", j, v[j]);
        //printf("beta = %.8f\n", beta);

        // update M: M[i:, i:] = M[i:, i:] - beta * vv^T * M[i:, i:]
        // v^T * M[i:, i:]
        for(int j = i; j < n; ++j) {
            temp[j - i] = 0.;
            for (int k = i; k < m; ++k) {
                temp[j - i] += v[k - i] * M[k + j * m];
            }
        }
        for (int j = i; j < n; ++j) {
            for (int k = i; k < m; ++k) {
                M[k + j * m] -= beta * v[k - i] * temp[j - i];
            }
        }

        if (i <= n - 2) {
            // Householder transformation to row; keep superdiagonal element
            house_vec(M + (i + 1) * m + i, n - 1 -i, m, v, & beta);
            // update M: M[i:, i+1:] = M[i:, i+1:] - beta * M[i:, i+1:] * vv^T
            // M[i:, i+1:] * v
            for (int j = i; j < m; ++j) {
                temp[j - i] = 0.;
                for (int k = i + 1; k < n; ++k) {
                    temp[j - i] += M[j + m * k] * v[k - i - 1];
                }
            }
            for (int k = i + 1; k < n; ++k) {
                for (int j = i; j < m; ++j) {
                    M[j + k * m] -= beta * v[k - i - 1] * temp[j - i];
                }
            }
        }
    }

    // print bidiagonal form
    //for (int j = 0; j < n; ++j) {
    //    if (j < n - 1) printf("B[%d, %d] = %.8f, B[%d, %d] = %.8f \n", 
    //                          j, j, M[j + j * m], j, j + 1, M[j + (j + 1) * m]);
    //    else printf("B[%d, %d] = %.8f\n", j, j,  M[j + j * m]);
    //}

    /****
     * apply QR iteration algorithm to solve svd of bidiagonal matrix
     * Golub-Reinsch SVD
     ****/
    int max_iter = 200;    // maximum number of iterations
    int i;

    // TODO: free M and copy diagonal and superdiagonal of M to another array

    int p, q;
    double C[4];  // lower,right 2 x 2 submatrix of M[p:n-q-1, p:n-q-1]^T M[p:n-q-1, p:n-q-1]
    double c_1, c_2, denom, y, z;
    for (i = 0; i < max_iter; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            // zero superdiagonal element that reaches machine precision
            if (fabs(M[(j + 1) * m + j]) <= EPS * (fabs(M[j * m + j]) 
                                                   + fabs(M[(j + 1) * m + j + 1]))) {
                M[(j + 1) * m + j] = 0.;
            }
        } 

        // smallest p and largest q that M[p+1:n-q, p+1:n-q] has no zero superdiagonal
        // M[n-q+1:n, n-q+1:n] diagonal
        q = 1;
        for (int j = n - 2; j >= 0; --j) {
            if (fabs(M[(j + 1) * m + j]) < EPS) q++;
            else break;
        }

        if (q == n) break;  // M is already diagonal

        q--;
        p = n - q - 1;
        for (int j = p - 1; j >= 0; --j) {
            if (fabs(M[(j + 1) * m + j]) >= EPS) p--;
            else break;
        }

        int decouple_flag = 0;
        // decoupling the problem
        // if zero in diagonal, zero out superdiagonal in the same row by Givens transformation
        for (int j = p; j < n - q; ++j) {
            if (M[j * m + j] < EPS) {
                decouple_flag = 1;

                if (j < n - q - 1) {
                    // Givens transformation for rows
                    for (int k = j + 1; k < n - q; ++k) {
                        denom = sqrt(M[j + k * m] * M[j + k * m] 
                                            + M[k + k * m] * M[k + k * m]);
                        c_1 = M[k + k * m] / denom;
                        c_2 = - M[j + k * m] / denom;

                        M[j + k * m] = 0.;
                        M[j + (k + 1) * m] = c_2 * M[k + (k + 1) * m];
                        M[k + k * m] = denom;
                        M[k + (k + 1) * m] *= c_1;
                    }
                } else {  // M[n-q-1, n-q-1] = 0.
                    // Givens transformation for columns 
                    for (int k = j - 1; k >= p; --k) {
                        denom = sqrt(M[k + k * m] * M[k + k * m] +
                                     M[k + j * m] * M[k + j * m]);

                        c_1 = M[k + k * m] / denom;
                        c_2 = M[k + j * m] / denom;

                        M[k - 1 + j * m] = -c_2 * M[k - 1 + k * m];
                        M[k + j * m] = 0.;
                        M[k + k * m] = denom;
                        M[k - 1 + k * m] *= c_1;
                    }
                }
            }
        }

        if (decouple_flag > 0) continue;  // repartition
        else {
            // Golub-Kahan SVD step on M[p:n-q-1, p:n-q-1]
            int ul = p;  // up left index of M[p:n-q-1, p:n-q-1]
            int br = n - q - 1;  // bottom right index of M[p:n-q-1, p:n-q-1]

            // set Wilkinson shift
            C[0] = M[br - 1 + (br - 1) * m] * M[br - 1 + (br - 1) * m];
            if (br - ul + 1 > 2) C[0] += M[br - 2 + (br - 1) * m] * M[br - 2 + (br - 1) * m];
            C[1] = M[br - 1 + (br - 1) * m] * M[br - 1 + br * m]; C[2] = C[1];
            C[3] = M[br + br * m] * M[br + br * m] + M[br * m + br - 1] * M[br * m + br - 1];
            double mu = wilkinson_shift(C);

            // chase the pulse
            y = M[ul + ul * m] * M[ul + ul * m] - mu;
            z = M[ul + (ul + 1) * m] * M[ul + ul * m];
            for (int j = ul; j < br; ++j) {
                // [y, z][c_1, -c_2; c_2, c_1] = [sqrt(y^2 + z^2), 0]
                denom = sqrt(y * y + z * z);
                c_1 = y / denom;
                c_2 = z / denom;

                // apply Givens on column j and j + 1
                double Mjj = M[j + j * m];
                M[j + j * m] = c_1 * Mjj + c_2 * M[j + (j + 1) * m];
                M[j + (j + 1) * m] *= c_1; M[j + (j + 1) * m] -= c_2 * Mjj;
                M[j + 1 + j * m] = c_2 * M[j + 1 + (j + 1) * m];
                M[j + 1 + (j + 1) * m] *= c_1;

                if (j > ul) {
                    M[j - 1 + j * m] = denom;
                    M[j - 1 + (j + 1) * m] = 0.;
                }

                // [c_1, c_2; -c_2, c_1] [y; z] = [sqrt; 0]
                y = M[j + j * m]; z = M[j + 1 + j * m];
                denom = sqrt(y * y + z * z);
                c_1 = y/ denom;
                c_2 = z/ denom;

                // apply on column 
                M[j + j * m] = denom;
                M[j + 1 + j * m] = 0.;

                Mjj = M[j + (j + 1) * m];  // actually M[j,j+1]
                M[j + (j + 1 ) * m] = c_1 * Mjj + c_2 * M[j + 1 + (j + 1) * m];
                M[j + 1 + (j + 1) * m] *= c_1; M[j + 1 + (j + 1) * m] -= c_2 * Mjj;

                if (j < br - 1) {
                    M[j + (j + 2) * m] = c_2 * M[j + 1 + (j + 2) * m];
                    M[j + 1 + (j + 2) * m] *= c_1 ;
                    y = M[j + (j + 1) * m];
                    z = M[j + (j + 2) * m];
                }
            }
        }
    }

    // the algorithm does not converge in max_iter iterations
    if (i >= max_iter) {
        printf("The algorithm computing SVD failed to converge.\n");
        free(v);
        free(M);
        free(temp);

        exit(1);
    }

    for (i = 0; i < num_sings; ++i) 
        *nuc += M[i * m + i];

    free(v);
    free(M);
    free(temp);
}

// compute Householder vector v such that (I - beta vv^T)a = +(-) |a| e_1
// a: pointer to first element in input array
// n: elements of a
// stride: distance between two elements in a
// v: Householder vector
// beta: coefficient of Householder vector, I - beta*v*v^T
void house_vec(double* a, int n, int stride, double* v, double* beta) {
    // sqaure sum except first element in a
    double squa_except_first = 0.;
    for (int i = 1; i < n; ++i) {
        squa_except_first += a[i * stride] * a[i * stride];
    }

    v[0] = 1.;
    for (int i = 1; i < n; ++i) {
        v[i] = a[i * stride];
    }

    if (squa_except_first < EPS) {
        *beta = 0.;
    } else {
        // norm of a
        double norm_a = sqrt(squa_except_first + a[0] * a[0]);

        if (a[0] <= 0)
            v[0] = a[0] - norm_a;
        else  // avoid cancellation if array is near e_1
            v[0] = -squa_except_first / (a[0] + norm_a);

        //printf("v[0]: %.8f\n", v[0]);
        *beta = 2. * (v[0] * v[0]) / (squa_except_first + v[0] * v[0]);
        
        for (int i = 1; i < n; ++i) {
            v[i] = v[i] / v[0];
        }
        v[0] = 1.;
    }
}

// Wilkinson shift (eigenvalue of 2x2 matrix that is nearer to C_22
double wilkinson_shift(const double C[]) {
    // [a, b; c, d] eigenvalues: root of x^2 - (a+d)x +(ad - bc) = 0
    // (a+d)/2 +\pm sqrt(bc + (a-d)^2/4)
    double mean = (C[0] + C[3]) / 2;
    double delta = sqrt(C[1] * C[2] + (C[0] - C[3]) * (C[0] - C[3]) / 4.);
    double lamb1 = mean + delta;
    double lamb2 = mean - lamb2;
    
    return fabs(lamb1 - C[3]) <= fabs(lamb2 - C[3])? lamb1 : lamb2;
}
