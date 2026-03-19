#include<iostream>
#include<mpi.h>
#include<iomanip>

using namespace std;

const int N = 5;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double A[N * N], B[N * N], C[N * N];
    cout<<size<<endl;
    if(rank == 0){
        for(int i = 0; i < N * N; i++){
            A[i] = i + 1;
        }
    }
    else if(rank == 1){
        for(int i = 0; i < N * N; i++){
            B[i] = i + 1;
        }
        MPI_Send(B, N * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        MPI_Recv(B, N * N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    MPI_Finalize();
}