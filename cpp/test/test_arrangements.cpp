#include "gtest/gtest.h"
#include <iostream>
#include <contact_modes/geometry/arrangements.hpp>


TEST(ARRANGEMENTS, INITIAL) {
    int n = 4;
    int d = 4;
    Eigen::MatrixXd A(n,d);
    Eigen::VectorXd b(n);
    A.setRandom();
    b.setRandom();

    IncidenceGraphPtr I = initial_arrangement(A, b, 1e-8);
    I->update_sign_vectors(1e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}