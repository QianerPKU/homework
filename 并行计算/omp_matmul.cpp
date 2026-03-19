#include<iostream>
#include<omp.h>
#include<iomanip>

const int N = 5;

using namespace std;
int main(){
    double A[N * N], B[N * N], C[N * N];
    for(int i = 0; i < N * N; i++){
        A[i] = i + 1;
        B[i] = i + 1;
    }

    omp_set_num_threads(4);

    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            C[i * N + j] = 0;
            for(int k = 0; k < N; k++){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    cout << "Result of A x B:" << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << setw(5) << C[i * N + j] << " ";
        }
        cout << endl;
    }
}