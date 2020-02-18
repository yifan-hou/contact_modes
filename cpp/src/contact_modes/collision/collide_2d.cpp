#include <contact_modes/collision/collide_2d.hpp>
#define CUTE_C2_IMPLEMENTATION
#include <contact_modes/extern/cute_c2.h>


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
        
    }

    c2Manifold m;
    c2PolytoPolyManifold(&A, nullptr, &B, nullptr, &m);

    Manifold2DPtr manifold = std::make_shared<Manifold2D>();
    manifold->pts_A.resize(m.count);
    manifold->pts_B.resize(m.count);
    manifold->dists.resize(m.count);
    for (int i = 0; i < m.count; i++) {
        manifold->dists[i] = m.depths[i];
        manifold->pts_A[i].x() = m.contact_points[i].x;
        manifold->pts_A[i].y() = m.contact_points[i].y;
    }
    manifold->normal.x() = m.n.x;
    manifold->normal.y() = m.n.y;

    return std::make_shared<Manifold2D>();
}