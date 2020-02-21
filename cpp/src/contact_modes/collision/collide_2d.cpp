#include <contact_modes/collision/collide_2d.hpp>
#define CUTE_C2_IMPLEMENTATION
#include <contact_modes/extern/cute_c2.h>
#include <math.h>
#include <iostream>

// int DEBUG=0;

Manifold2DPtr collide_2d(const std::vector<Eigen::Vector2d>& verts_A,
                         const std::vector<Eigen::Vector2d>& verts_B,
                         const Eigen::Vector3d& xi_A,
                         const Eigen::Vector3d& xi_B) {
    //
    assert(verts_A.size() <= 8);
    assert(verts_B.size() <= 8);

    c2Poly A, B;
    A.count = verts_A.size();
    for (int i = 0; i < A.count; i++) {
        A.verts[i].x = cos(xi_A(2)) * verts_A[i](0) - sin(xi_A(2)) * verts_A[i](1) + xi_A(0);
        A.verts[i].y = sin(xi_A(2)) * verts_A[i](0) + cos(xi_A(2)) * verts_A[i](1) + xi_A(1);
    }
    B.count = verts_B.size();
    for (int i = 0; i < B.count; i++) {
        B.verts[i].x = cos(xi_B(2)) * verts_B[i](0) - sin(xi_B(2)) * verts_B[i](1) + xi_B(0);
        B.verts[i].y = sin(xi_B(2)) * verts_B[i](0) + cos(xi_B(2)) * verts_B[i](1) + xi_B(1);
    }
    c2MakePoly(&A);
    c2MakePoly(&B);

    c2Manifold m;
    c2PolytoPolyManifold(&A, nullptr, &B, nullptr, &m);
    // std::cout << m.count << std::endl;

    Manifold2DPtr manifold = std::make_shared<Manifold2D>();

    manifold->pts_A.resize(m.count);
    manifold->pts_B.resize(m.count);
    manifold->dists.resize(m.count);
    for (int i = 0; i < m.count; i++) {
        manifold->dists[i] = m.depths[i];
        if (1) {
            std::cout << "contact: " << "[" << m.contact_points[i].x << ", " << m.contact_points[i].y << "]" << std::endl;
            std::cout << " normal: " << "[" << m.n.x << ", " << m.n.y << "]" << std::endl;
            std::cout << "  depth: " << m.depths[i] << std::endl;
        }

        // std::cout << m.depths[i] << std::endl;
        Eigen::Vector2d& a = manifold->pts_A[i];
        Eigen::Vector2d& b = manifold->pts_B[i];
        a.x() = m.contact_points[i].x;
        b.x() = m.contact_points[i].x + m.depths[i] * m.n.x;
        a.y() = m.contact_points[i].y;
        b.y() = m.contact_points[i].y + m.depths[i] * m.n.y;

        // manifold->pts_A[i].x() = (m.contact_points[i].x- xi_A(0))*cos(xi_A(2)) + (m.contact_points[i].y- xi_A(1))*sin(xi_A(2));
        // manifold->pts_A[i].y() = (m.contact_points[i].y- xi_A(1))*cos(xi_A(2)) + (-m.contact_points[i].x + xi_A(0))*sin(xi_A(2));
        // manifold->pts_B[i].x() = (m.contact_points[i].x- xi_B(0))*cos(xi_B(2)) + (m.contact_points[i].y- xi_B(1))*sin(xi_B(2));
        // manifold->pts_B[i].y() = (m.contact_points[i].y- xi_B(1))*cos(xi_B(2)) + (-m.contact_points[i].x + xi_B(0))*sin(xi_B(2));
    }
    manifold->normal.x() = m.n.x;
    manifold->normal.y() = m.n.y;

    return manifold;
}
