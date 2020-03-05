#include <contact_modes/geometry/arrangements.hpp>
#include <iostream>

// #define DEBUG
// #define PROFILE

int DEBUG = 0;
int PROFILE = 0;

int get_sign(double v, double eps) {
    assert(eps > 0);
    if (v > eps) {
        return 1;
    } else if (v < -eps) {
        return -1;
    } else {
        return 0;
    }
}

void get_sign(const Eigen::VectorXd& v, Eigen::VectorXi& sv, double eps) {
    eps = abs(eps);
    sv.resize(v.size());
    for (int i = 0; i < v.size(); i++) {
        sv[i] = get_sign(v[i], eps);
    }
}

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
    if (e->subfaces.size() == 2) {
        auto it = e->subfaces.begin();
        NodePtr v0 = *it++;
        NodePtr v1 = *it;
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
        NodePtr v0 = *e->subfaces.begin();
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
    // Assert we are given d linearly independent hyperplanes.
    int n = A.rows();
    int d = A.cols();
    assert(n == d);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    if (DEBUG) {
        std::cout << "rank(A) " << qr.rank() << std::endl;
    }
    assert(qr.rank() == d);

    // Build faces from top to bottom.
    IncidenceGraphPtr I = std::make_shared<IncidenceGraph>(d);
    I->A = A;
    I->b = b;
    // d+1 face
    NodePtr one = std::make_shared<Node>(d+1);
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
        NodePtr f = std::make_shared<Node>(d);
        f->_sv_key = sv;
        f->superfaces.insert(one);
        one->subfaces.insert(f);
        I->add_node(f);
    }
    // k faces, 0 <= k <= d-1
    for (int k = d-1; k >= 0; k--) {
        // std::cout << "k " << k << std::endl;
        auto iter = I->rank(k + 1).begin();
        auto end = I->rank(k + 1).end();
        while (iter != end) {
            NodePtr g = iter->second;
            // std::cout << "g " << g->_sv_key << std::endl;
            sv = g->_sv_key;
            assert(sv.size() == d);
            for (int i = 0; i < d; i++) {
                if (sv[i] == '0') {
                    continue;
                }
                char tmp = sv[i];
                sv[i] = '0';
                NodePtr f = I->get_node(sv, k);
                if (!f) {
                    f = std::make_shared<Node>(k);
                    f->_sv_key = sv;
                    I->add_node(f);
                }
                f->superfaces.insert(g);
                g->subfaces.insert(f);
                sv[i] = tmp;
            }
            iter++;
        }
    }
    // -1 face
    NodePtr zero = std::make_shared<Node>(-1);
    NodePtr v = I->get_node(std::string(d, '0'), 0);
    zero->superfaces.insert(v);
    v->subfaces.insert(zero);
    I->add_node(zero);

    // Compute interior point for 0 face.
    v->interior_point = qr.solve(b);

    // Compute interior point for 1 faces.
    {
        auto iter = I->rank(1).begin();
        auto end = I->rank(1).end();
        while (iter != end) {
            NodePtr e = iter->second;
            int s0 = e->_sv_key.find('+');
            int s1 = e->_sv_key.find('-');
            if (s0 >= 0) {
                e->interior_point = v->interior_point + I->A.row(s0).transpose();
            }
            if (s1 >= 0) {
                e->interior_point = v->interior_point - I->A.row(s1).transpose();
            }
            assert((s0+1)*(s1+1) == 0);
            iter++;
        }
    }

    #ifdef EIGEN_VECTORIZE
    std::cout << "VECTORIZED" << std::endl;
    #endif

    // Compute interior point for k faces.
    for (int k = 2; k < d + 1; k++) {
        auto iter = I->rank(k).begin();
        auto end = I->rank(k).end();
        
        while (iter != end) {
            NodePtr f = iter->second;
            f->interior_point.setZero(d);
            int i = 1;
            for (auto u : f->subfaces) {
                f->interior_point += u->interior_point;
            }
            // std::cout << f->subfaces.size() << std::endl;
            f->interior_point /= f->subfaces.size();
            iter++;
        }
    }

    return I;
}

void increment_arrangement(
    const Eigen::VectorXd& a, double b, 
    IncidenceGraphPtr I, double eps) {
    // 
}