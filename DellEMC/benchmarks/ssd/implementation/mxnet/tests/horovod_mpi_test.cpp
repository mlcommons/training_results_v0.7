#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>

#define NWARMUP 1000
#define NTRIALS 10000

int main() {

  // Initialize MPI with thread support like Horovod
  int mpi_threads_required = MPI_THREAD_MULTIPLE;
  int mpi_threads_provided;
  MPI_Init_thread(nullptr, nullptr, mpi_threads_required,
                  &mpi_threads_provided);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  // Randomly initialize vector of 2 long long (used for Horovod coordination)
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::uniform_int_distribution<long long> dist;
  std::vector<long long> vec(2);
  vec[0] = dist(rng);
  vec[1] = dist(rng);

  // Warmup
  if (comm_rank == 0) std::cout << "Running " << NWARMUP << " warmup MPI_Allreduce trials..." << std::endl;
  for (int i = 0; i < NWARMUP; i++) {
    int ret_code = MPI_Allreduce(MPI_IN_PLACE, vec.data(), vec.size(),
                                 MPI_LONG_LONG_INT, MPI_BAND, MPI_COMM_WORLD);
  }

  // Trials
  if (comm_rank == 0) std::cout << "Running " << NTRIALS << " MPI_Allreduce trials..." << std::endl;
  double latency_max = 0;
  double latency_min = 1000000;
  double duration_ms_trials = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < NTRIALS; i++) {
    auto tst = std::chrono::high_resolution_clock::now();
    int ret_code = MPI_Allreduce(MPI_IN_PLACE, vec.data(), vec.size(),
                                 MPI_LONG_LONG_INT, MPI_BAND, MPI_COMM_WORLD);
    auto tet = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(tet - tst);
    double duration_trial_ms = duration.count() / 1000000.0;
    latency_min = std::min(latency_min, duration_trial_ms);
    latency_max = std::max(latency_max, duration_trial_ms);
    duration_ms_trials += duration_trial_ms;
  }

  double latency_avg = duration_ms_trials / NTRIALS;

  // Reduce min, max, and average latencies across all workers and trials
  double latency_min_global, latency_max_global, latency_avg_global;
  MPI_Reduce(&latency_min, &latency_min_global, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&latency_max, &latency_max_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&latency_avg, &latency_avg_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (comm_rank == 0) {
    std::cout << "global MIN latency: " << latency_min_global << " ms" << std::endl;
    std::cout << "global MAX latency: "  << latency_max_global << " ms" << std::endl;
    std::cout << "global AVG latency: " << latency_avg_global / comm_size << " ms" << std::endl;
  }

  MPI_Finalize();
}
