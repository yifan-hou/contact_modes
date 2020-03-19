#include <contact_modes/geometry/arrangements.hpp>
#include <chrono>
#include <iostream>


int main() {
    // Set random seed.
    srand(0);

    // Benchmark initial arrangements.
    for (int n = 5; n < 10; n++) {
        Eigen::MatrixXd A(n,n);
        Eigen::VectorXd b(n);

        A.setRandom();
        b.setRandom();
        auto start = std::chrono::high_resolution_clock::now();
        IncidenceGraphPtr I = initial_arrangement(A, b, 1e-8);
        auto end = std::chrono::high_resolution_clock::now();

        // std::cout << n << " " << 
        // std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
        // << " ms" << std::endl;
    }
}