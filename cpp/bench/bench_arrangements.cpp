#include <contact_modes/geometry/arrangements.hpp>
#include <chrono>
#include <iostream>


int main() {
    // Set random seed.
    srand(0);

    // // Benchmark initial arrangements.
    // for (int n = 5; n < 14; n++) {
    //     Eigen::MatrixXd A(n,n);
    //     Eigen::VectorXd b(n);

    //     A.setRandom();
    //     b.setRandom();
    //     auto start = std::chrono::high_resolution_clock::now();
    //     IncidenceGraphPtr I = initial_arrangement(A, b, 1e-8);
    //     auto end = std::chrono::high_resolution_clock::now();
    // }

    // Benchmark increment arrangements.
    for (int j = 0; j < 20; j++)
    {
        srand(0);

        int n = 12;
        int d = 11;
        Eigen::MatrixXd A(d,d);
        Eigen::VectorXd b(d);
        A.setRandom();
        b.setRandom();

        IncidenceGraphPtr I = initial_arrangement(A, b, 1e-8);
        // I->update_sign_vectors(1e-8);

        for (int i = d; i < n; i++) {
            Eigen::VectorXd a(d);
            Eigen::VectorXd b(1);
            a.setRandom();
            b.setRandom();
            increment_arrangement(a, b[0], I, 1e-8);
        }
    }
}