#include "gtest/gtest.h"
#include <iostream>
#include <contact_modes/geometry/arrangements.hpp>


TEST(ARRANGEMENTS, INITIAL) {
    return;
    for (int n = 10; n < 13; n++) {
        Eigen::MatrixXd A(n,n);
        Eigen::VectorXd b(n);
        A.setRandom();
        b.setRandom();

        IncidenceGraphPtr I = initial_arrangement(A, b, 1e-8);
        I->update_sign_vectors(1e-8);
    }
}

TEST(ARRANGEMENTS, INCREMENT) {
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}