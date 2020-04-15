#include "gtest/gtest.h"
#include <iostream>
#include <contact_modes/geometry/incidence_graph.hpp>


TEST(INCIDENCE_GRAPH, ARC_POOL) {
    boost::object_pool<Arc> p;
    
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}