#pragma once
#include <contact_modes/collision/manifold_2d.hpp>


Manifold2DPtr collide_2d(const std::vector<Eigen::Vector2d>& verts_A,
                         const std::vector<Eigen::Vector2d>& verts_B,
                         const Eigen::Vector3d& tf_A,
                         const Eigen::Vector3d& tf_B);