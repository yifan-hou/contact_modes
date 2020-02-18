#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>


struct Manifold2D {
    std::vector<Eigen::Vector2d> pts_A;
    std::vector<Eigen::Vector2d> pts_B;
    Eigen::Vector2d normal;
    std::vector<double> dists;
};

typedef std::shared_ptr<Manifold2D> Manifold2DPtr;