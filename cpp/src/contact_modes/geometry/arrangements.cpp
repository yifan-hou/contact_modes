#include <contact_modes/geometry/arrangements.hpp>
#include <chrono>
#include <iostream>


static int DEBUG=0;
static int PROFILE=1;

// TODO Replace me
int get_sign(NodePtr f, const Eigen::VectorXd& a, double b, double eps) {
    return get_sign(a.dot(f->interior_point) - b, eps);
}

void get_vector(NodePtr e, Eigen::VectorXd& v_e) {
    assert(e->rank == 1);
    IncidenceGraphPtr I = e->_graph;
    v_e.setZero();
    for (int u_id : e->subfaces) {
        NodePtr u = I->_nodes[u_id];
        v_e += u->interior_point;
    }
    if (e->subfaces.size() == 2) {
        v_e /= 2.0;
    } else if (e->subfaces.size() == 1) {
        v_e = e->interior_point - v_e;
    } else {
        assert(false);
    }
}

int get_color_vertex(NodePtr v, const Eigen::VectorXd& a, double b, double eps) {
    assert(v->rank == 0);
    int s = get_sign(v, a, b, eps);
    if (s == 0) {
        return COLOR_AH_CRIMSON;
    } else {
        return COLOR_AH_WHITE;
    }
}

int get_color_edge(NodePtr e, const Eigen::VectorXd& a, double b, double eps) {
    assert(e->rank == 1);
    IncidenceGraphPtr I = e->_graph;
    if (e->subfaces.size() == 2) {
        auto it = e->subfaces.begin();
        NodePtr v0 = I->node(*it++);
        NodePtr v1 = I->node(*it);
        assert(v0 != v1);
        int s0 = get_sign(v0, a, b, eps);
        int s1 = get_sign(v1, a, b, eps);
        if (s0 * s1 == 1) {
            return COLOR_AH_WHITE;
        } else if (s0 == 0 && s1 == 0) {
            return COLOR_AH_CRIMSON;
        } else if (s0 == 0 || s1 == 0) {
            return COLOR_AH_PINK;
        } else if (s0 + s1 == 0) {
            return COLOR_AH_RED;
        } else {
            assert(false);
        }
    } else if (e->subfaces.size() == 1) {
        NodePtr v0 = I->node(*e->subfaces.begin());
        int s0 = get_sign(v0, a, b, eps);
        Eigen::VectorXd v_e(e->interior_point.size());
        int s_e = get_sign(a.dot(v_e), eps);
        if (s0 == 0 && s_e == 0) {
            return COLOR_AH_CRIMSON;
        } else if (s0 == 0 && s_e != 1) {
            return COLOR_AH_PINK;
        } else if  (s0 * s_e == 1) {
            return COLOR_AH_WHITE;
        } else if (s0 * s_e == -1) {
            return COLOR_AH_RED;
        } else if (s0 != 0 and s_e == 0) {
            return COLOR_AH_WHITE;
        } else {
            assert(false);
        }
    } else {
        assert(false);
    }
    return COLOR_AH_WHITE;
}

IncidenceGraphPtr initial_arrangement(
    const Eigen::MatrixXd& A, 
    const Eigen::VectorXd& b, 
    double eps) {

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_total;
    if (PROFILE) {
        start = std::chrono::high_resolution_clock::now();
        start_total = std::chrono::high_resolution_clock::now();
    }
    
    // Assert we are given d linearly independent hyperplanes.
    int n = A.rows();
    int d = A.cols();
    assert(n == d);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    if (DEBUG) {
        std::cout << "rank(A) " << qr.rank() << std::endl;
    }
    assert(qr.rank() == d);

    if (PROFILE) {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  init: " << n << "x" << n << std::endl;
        std::cout << "  rank: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
        << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
    }

    // Build faces from top to bottom.
    IncidenceGraphPtr I = std::make_shared<IncidenceGraph>(d);
    I->A = A;
    I->b = b;
    // d+1 face
    NodePtr one = I->make_node(d+1);
    I->add_node(one);
    // d faces
    int num_d_faces = 1 << d;
    std::string sv(d, '+');
    for (int i = 0; i < num_d_faces; i++) {
        // Create sign vector string.
        for (int j = 0; j < d; j++) {
            if (i & 1 << j) {
                sv[j] = '+';
            } else {
                sv[j] = '-';
            }
        }
        if (DEBUG) {
            std::cout << sv << std::endl;
        }
        // Create cell.
        NodePtr f = I->make_node(d);
        f->_key = sv;
        f->superfaces.insert(one->_id);
        one->subfaces.insert(f->_id);
        I->add_node(f);
    }
    // k faces, 0 <= k <= d-1
    for (int k = d-1; k >= 0; k--) {
        if (DEBUG) {
            std::cout << "k " << k << std::endl;
        }
        auto iter = I->rank(k + 1).begin();
        auto end  = I->rank(k + 1).end();
        while (iter != end) {
            NodePtr g = I->node(iter->second);
            if (DEBUG) {
                std::cout << "g " << g->_key << std::endl;
            }
            sv = g->_key;
            assert(sv.size() == d);
            for (int i = 0; i < d; i++) {
                if (sv[i] == '0') {
                    continue;
                }
                char tmp = sv[i];
                sv[i] = '0';
                NodePtr f = I->get_node(sv, k);
                if (!f) {
                    f = I->make_node(k);
                    f->_key = sv;
                    I->add_node(f);
                }
                f->superfaces.insert(g->_id);
                g->subfaces.insert(f->_id);
                sv[i] = tmp;
            }
            iter++;
        }
    }
    // -1 face
    NodePtr zero = I->make_node(-1);
    NodePtr v = I->get_node(std::string(d, '0'), 0);
    zero->superfaces.insert(v->_id);
    v->subfaces.insert(zero->_id);
    I->add_node(zero);

    if (PROFILE) {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " faces: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
        << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
    }
    
    // Compute interior point for 0 face.
    v->update_interior_point(eps);

    // Compute interior point for 1 faces.
    {
        auto iter = I->rank(1).begin();
        auto  end = I->rank(1).end();
        while (iter != end) {
            NodePtr e = I->node(iter->second);
            e->update_interior_point(eps);
            iter++;
        }
    }

    // Compute interior point for k faces.
    for (int k = 2; k < d + 1; k++) {
        auto iter = I->rank(k).begin();
        auto end = I->rank(k).end();
        while (iter != end) {
            NodePtr f = I->node(iter->second);
            f->update_interior_point(eps);
            iter++;
        }
    }

    if (PROFILE) {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "int pt: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
        << " ms" << std::endl;

        std::cout << " total: " << 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_total).count() / 1e6
        << " ms" << std::endl;
    }

    return I;
}

void increment_arrangement(const Eigen::VectorXd& a, double b, 
                           IncidenceGraphPtr I, double eps) {
    // 
    
}