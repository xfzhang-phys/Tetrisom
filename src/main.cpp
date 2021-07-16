#include <iostream>
#include <string>
#include <chrono>
#include <mpi.h>
#include "som.h"

using namespace std::chrono;

int main(int argc, char **argv) {

    // MPI
    int mpi_size = 1;
    int mpi_rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // timer
    auto start = steady_clock::now();

    // random number generator
    std::random_device r;
    std::mt19937_64 rng(r());

    SOM sw(10, 65, 100, 50, 60, 200, 0.001, 0.001, 2., 2., 20., 0., 1.5, 0.5, 1.0, true, "imtime");

    // loop for attempts
    for (uint32_t il = 0; il < sw.Lmax; il++) {
        sw.attempt(il, &rng);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // outputs
    sw.output(mpi_rank, mpi_size);

    // timer
    auto end = steady_clock::now();
    auto elapsed_time = duration<double>(end - start);
    if (mpi_rank == 0) {
        std::cout << "Elapsed time: " << elapsed_time.count() << " s." << std::endl;
    }

    MPI_Finalize();
}
