#include <contact_modes/geometry/arrangements.hpp>
#include <chrono>
#include <iostream>
#include <list>


static int DEBUG=0;
static int PROFILE=1;

// TODO Replace me
int get_position(NodePtr f, const Eigen::VectorXd& a, double b, double eps) {
    return get_position(a.dot(f->interior_point) - b, eps);
}

void get_vector(NodePtr e, Eigen::VectorXd& v_e) {
    assert(e->rank == 1);
    IncidenceGraphPtr I = e->_graph;
    v_e.resize(I->A.cols());
    v_e.setZero();
    for (int u_id : e->subfaces) {
        NodePtr u = I->node(u_id);
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
    int s = get_position(v, a, b, eps);
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
        int s0 = get_position(v0, a, b, eps);
        int s1 = get_position(v1, a, b, eps);
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
        int s0 = get_position(v0, a, b, eps);
        Eigen::VectorXd v_e(e->interior_point.size());
        get_vector(e, v_e);
        int s_e = get_position(a.dot(v_e), eps);
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

int sign_vector_to_base3(std::string sv) {
    int n = 0;
    int b = 1;
    for (int i = 0; i < sv.size(); i++) {
        if (sv[i] == '0') {
            n += 0 * b;
        } else if (sv[i] == '+') {
            n += 1 * b;
        } else if (sv[i] == '-') {
            n += 2 * b;
        } else {
            assert(false);
        }
        b *= 3;
    }
    return n;
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
        // std::cout << "rank(A) " << qr.rank() << std::endl;
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

    // Allocate k-faces, 0 <= k <= d.
    IncidenceGraphPtr I = std::make_shared<IncidenceGraph>(d);
    I->_nodes.resize((int) pow(3, d));
    for (int i = 0; i < I->_nodes.size(); i++) {
        NodePtr node = std::make_shared<Node>(-10);
        node->_id = i;
        node->_graph = I;
        I->_nodes[i] = node;
    }
    // Copy halfspaces.
    I->A = A;
    I->b = b;

    // We build faces from top to bottom, starting with the d+1 face.
    NodePtr one = I->make_node(d+1);
    I->add_node_to_rank(one);
    // Next we build the regions, i.e. d faces.
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
            // std::cout << sv << std::endl;
        }
        // Create region.
        NodePtr f = I->_nodes[sign_vector_to_base3(sv)];
        f->rank = d;
        f->_key = sv;
        I->add_node_to_rank(f);
        I->add_arc(f, one);
    }
    // k faces, 0 <= k <= d-1
    for (int k = d-1; k >= 0; k--) {
        for (int i_g : I->rank(k + 1)) {
            NodePtr g = I->node(i_g);
            sv = g->_key;
            assert(sv.size() == d);
            for (int i = 0; i < d; i++) {
                if (sv[i] == '0') {
                    continue;
                }
                char tmp = sv[i];
                sv[i] = '0';
                NodePtr f = I->_nodes[sign_vector_to_base3(sv)];
                f->rank = k;
                f->_key = sv;
                if (f->_black_bit == 0) {
                    I->add_node_to_rank(f);
                    f->_black_bit = 1;
                }
                I->add_arc(f, g);
                sv[i] = tmp;
            }
        }
    }
    // -1 face
    NodePtr zero = I->make_node(-1);
    NodePtr v = I->node(I->rank(0)[0]);
    I->add_arc(zero, v);
    I->add_node_to_rank(zero);

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
        for (int i_e : I->rank(1)) {
            NodePtr e = I->node(i_e);
            e->update_interior_point(eps);
        }
    }

    // Compute interior point for k faces.
    for (int k = 2; k < d + 1; k++) {
        for (int i_f : I->rank(k)) {
            NodePtr f = I->node(i_f);
            f->update_interior_point(eps);
        }
    }

    // Reset black bit.
    for (int i = 0; i < I->_nodes.size(); i++) {
        I->_nodes[i]->_black_bit = 0;
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

void increment_arrangement(Eigen::VectorXd a, double b, 
                           IncidenceGraphPtr I, double eps) {
    // Normalize halfspace, |a| = 1.
    double norm_a = a.norm();
    a /= norm_a;
    b /= norm_a;
    I->add_hyperplane(a, b);

    // =========================================================================
    // Phase 1: Find an edge e₀ in 𝓐(H) such that cl(e₀) ∩ h ≠ ∅
    // =========================================================================

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_total;
    if (PROFILE) {
        std::cout << "   incr: " << I->A.rows() << "x" << I->A.cols() << std::endl;
        std::cout << "phase 1: " << std::endl;
        start = std::chrono::high_resolution_clock::now();
    }
    int n = I->A.rows();
    NodePtr v = I->node(I->rank(0)[0]);
    // Find an incident edge e on v such that aff(e) is not parallel to h.
    NodePtr e;
    Eigen::VectorXd v_e;
    double dist;
    for (int i : v->superfaces) {
        e = I->node(i);
        get_vector(e, v_e);
        dist = a.dot(v_e);
        if (dist > eps) {
            break;
        }
    }
    if (DEBUG) {
        std::cout << "\tdot: " << dist << std::endl;
        assert(dist > eps);
    }
    // Find edge e₀ such that cl(e₀) ∩ h ≠ ∅.
    NodePtr e0 = e;
    Eigen::VectorXd v_e0 = v_e;
    while (true) {
        if (get_color_edge(e0, a, b, eps) > COLOR_AH_WHITE) {
            break;
        }
        // Find v(e0) closer to h.
        NodePtr v;
        if (e0->subfaces.size() == 2) {
            auto i = e0->subfaces.begin();
            NodePtr v0 = I->node(*i++);
            NodePtr v1 = I->node(*i);
            double d0 = abs(a.dot(v0->interior_point) - b);
            double d1 = abs(a.dot(v1->interior_point) - b);
            if (d0 < d1) {
                v = v0;
            } else {
                v = v1;
            }
        } else if (e0->subfaces.size() == 1) {
            v = I->node(*e0->subfaces.begin());
        } else {
            assert(false);
        }
        // Find e in v such that aff(e0) == aff(e).
        NodePtr e_min;
        Eigen::VectorXd v_min;
        double min_dist = std::numeric_limits<float>::infinity();
        for (int i_e : v->superfaces) {
            e = I->node(i_e);
            if (e == e0) {
                continue;
            }
            get_vector(e, v_e);
            v_e.normalize();
            double dist = (v_e - v_e0.dot(v_e) * v_e).norm();
            if (dist < min_dist) {
                e_min = e;
                v_min = v_e;
                min_dist = dist;
            }
        }
        e0 = e_min;
        if (DEBUG) {
            std::cout << " e min: " << e_min->_key << std::endl;
            std::cout << " d min: " << min_dist << std::endl;
        }
    }
    if (DEBUG) {
        std::cout << "\te0.color: " << 
            get_color_ah_string(get_color_edge(e0, a, b, eps)) << std::endl;
        std::cout << "\te0.key  : " << e0->_key << std::endl;
    }

    // =========================================================================
    // Phase 2: Mark all faces f with cl(f) ∩ h ≠ ∅ pink, red, or crimson.
    // =========================================================================

    int mark_count = 0;
    if (PROFILE) {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  total: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
        << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        std::cout << "phase 2: " << std::endl;
    }
    // Add some 2 face incident upon e₀ to Q and mark it green.
    NodePtr f = I->node(*e0->superfaces.begin());
    f->_color = COLOR_AH_GREEN;
    std::list<NodePtr> Q;
    Q.push_back(f);
    // Color vertices, edges, and 2 faces of 𝓐(H).
    int d = a.size();
    std::vector<std::vector<NodePtr> > L(d+1);
    for (int k = 0; k < d+1; k++) {
        L[k].clear();
    }
    NodePtr g;
    while (!Q.empty()) {
        f = Q.front();
        Q.pop_front();
        for (int i_e : f->subfaces) {
            e = I->node(i_e);
            if (e->_color != COLOR_AH_WHITE) {
                continue;
            }
            int color_e = get_color_edge(e, a, b, eps);
            if (color_e > COLOR_AH_WHITE) {
                // Mark each white vertex v ∈ h crimson and insert v into L₀.
                for (int i_v : e->subfaces) {
                    v = I->node(i_v);
                    if (v->_color == COLOR_AH_WHITE) {
                        int color_v = get_color_vertex(v, a, b, eps);
                        if (color_v == COLOR_AH_CRIMSON) {
                            v->_color = color_v;
                            L[0].push_back(v);
                        }
                    }
                }
                // Color e and insert e into L₁.
                e->_color = color_e;
                L[1].push_back(e);
                // Mark all white 2 faces green and put them into Q.
                for (int i_g : e->superfaces) {
                    g = I->node(i_g);
                    if (g->_color == COLOR_AH_WHITE) {
                        g->_color = COLOR_AH_GREEN;
                        Q.push_back(g);
                    }
                }
            }
        }
    }
    // Color k faces, 2 ≤ k ≤ d.
    for (int k = 2; k < d+1; k++) {
        for (NodePtr f : L[k-1]) {
            for (int i_g : f->superfaces) {
                g = I->node(i_g);
                if (g->_color != COLOR_AH_WHITE && g->_color != COLOR_AH_GREEN) {
                    continue;
                }
                if (f->_color == COLOR_AH_PINK) {
                    bool above = false;
                    bool below = false;
                    for (int i_fg : g->subfaces) {
                        NodePtr f_g = I->node(i_fg);
                        int s;
                        if (f_g->_color == COLOR_AH_RED) {
                            above = true;
                            below = true;
                            break;
                        }
                        if (f_g->_sign_bit_n != n) {
                            s = get_position(a.dot(f_g->interior_point) - b, eps);
                            f_g->_sign_bit_n = n;
                            f_g->_sign_bit = s;
                        } else {
                            s = f_g->_sign_bit;
                        }
                        if (s > 0) {
                            above = true;
                        } else if (s < 0) {
                            below = true;
                        }
                    }
                    if (above && below) {
                        g->_color = COLOR_AH_RED;
                    } else {
                        g->_color = COLOR_AH_PINK;
                    }
                }
                else if (f->_color == COLOR_AH_RED) {
                    g->_color = COLOR_AH_RED;
                }
                else if (f->_color == COLOR_AH_CRIMSON) {
                    bool crimson = true;
                    for (int i_fg : g->subfaces) {
                        NodePtr f_g = I->node(i_fg);
                        if (f_g->_color != COLOR_AH_CRIMSON) {
                            crimson = false;
                            break;
                        }
                    }
                    if (crimson) {
                        g->_color = COLOR_AH_CRIMSON;
                    } else {
                        g->_color = COLOR_AH_PINK;
                    }
                }
                else {
                    std::cout << get_color_ah_string(f->_color) << std::endl;
                    assert(false);
                }
                L[k].push_back(g);
            }
        }
    }

    // =========================================================================
    // Phase 3: Update all marked faces.
    // =========================================================================

    double time_step_1 = 0;
    double time_step_2 = 0;
    double time_step_3 = 0, time_step_3_copy = 0, time_step_3_ins = 0;
    double time_step_4 = 0;
    double time_step_5 = 0;
    double time_step_6 = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_step;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_1;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_2;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_3;
    if (PROFILE) {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  total: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
        << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        std::cout << "phase 3:" << std::endl;
    }


    for (int k = 0; k < d + 1; k++) {
        int n_Lk = L[k].size();
        for (int i = 0; i < n_Lk; i++) {
            NodePtr g = L[k][i];
            switch (g->_color) {

                case COLOR_AH_PINK: {
                g->_color = COLOR_AH_GREY;
                // Add to grey subfaces of superfaces.
                for (int i_u : g->superfaces) {
                    NodePtr u = I->node(i_u);
                    u->_grey_subfaces.push_back(g->_id);
                }
                break; }

                case COLOR_AH_CRIMSON: {
                g->_color = COLOR_AH_BLACK;
                // Add to black subfaces of superfaces.
                for (int i_u : g->superfaces) {
                    NodePtr u = I->node(i_u);
                    u->_black_subfaces.push_back(g->_id);
                }
                break; }
                
                case COLOR_AH_RED: {

                if (PROFILE) {
                    start_step = std::chrono::high_resolution_clock::now();
                }
                // Step 1. Create g_a = g ∩ h⁺ and g_b = g ∩ h⁻. Remove g from
                // 𝓐(H) and Lₖ and replace with g_a, g_b.
                g->update_sign_vector(eps);
                NodePtr g_a = g;
                g_a->_color = COLOR_AH_GREY;
                g_a->_key = g->sign_vector;
                g_a->_key.back() = '+';
                g_a->_graph = I;
                I->_nodes[g_a->_id] = g_a;
                NodePtr g_b = I->make_node(k);
                g_b->_color = COLOR_AH_GREY;
                g_b->_key = g->sign_vector;
                g_b->_key.back() = '-';
                I->add_node_to_rank(g_b);
                L[k].push_back(g_b);

                if (PROFILE) {
                    auto end_step = std::chrono::high_resolution_clock::now();
                    time_step_1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end_step - start_step).count() / 1e6;
                    start_step = std::chrono::high_resolution_clock::now();
                }
                // Step 2. Create the black face f = g ∩ h, connect it to g_a
                // and g_b, and put f into 𝓐(H) and Lₖ₋₁.
                NodePtr f = I->make_node(k-1);
                f->_color = COLOR_AH_BLACK;
                f->_key = g->sign_vector;
                f->_key.back() = '0';
                I->add_arc(f, g_a);
                I->add_arc(f, g_b);
                g_a->_black_subfaces = {f->_id};
                g_b->_black_subfaces = {f->_id};
                L[k-1].push_back(f);
                I->add_node_to_rank(f);

                if (PROFILE) {
                    auto end_step = std::chrono::high_resolution_clock::now();
                    time_step_2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end_step - start_step).count() / 1e6;
                    start_step = std::chrono::high_resolution_clock::now();
                }
                // Step 3. Connect each red superface of g with g_a and g_b.
                for (int i_r : g->superfaces) {
                    NodePtr r = I->node(i_r);
                    if (DEBUG) {
                        assert(r->_color == COLOR_AH_RED || r->rank == d+1);
                    }
                    I->add_arc(r, g_b);
                    r->_grey_subfaces.push_back(g_a->_id);
                    r->_grey_subfaces.push_back(g_b->_id);
                }

                if (PROFILE) {
                    auto end_step = std::chrono::high_resolution_clock::now();
                    time_step_3 += std::chrono::duration_cast<std::chrono::nanoseconds>(end_step - start_step).count() / 1e6;
                    start_step = std::chrono::high_resolution_clock::now();
                }
                // Step 4. Connect each white or grey subface of g with g_a if
                // it is in h⁺, and with g_b, otherwise.
                ArcListIterator iter = g->subfaces.begin();
                ArcListIterator end  = g->subfaces.end();
                while (iter != end) {
                    int i_u = *iter;
                    NodePtr u = I->node(i_u);
                    if (u->_color != COLOR_AH_WHITE && u->_color != COLOR_AH_GREY) {
                        if (DEBUG) {
                            assert(u->_color == COLOR_AH_BLACK);
                        }
                        continue;
                    }
                    int s;
                    if (u->_sign_bit_n != n) {
                        s = get_position(a.dot(u->interior_point) - b, eps);
                        u->_sign_bit_n = n;
                        u->_sign_bit = s;
                    } else {
                        s = u->_sign_bit;
                    }
                    if (s == 1) {
                        if (u->_color == COLOR_AH_GREY) {
                            g_a->_grey_subfaces.push_back(u->_id);
                        }
                    } else if (s == -1) {
                        if (u->_color == COLOR_AH_GREY) {
                            g_b->_grey_subfaces.push_back(u->_id);
                        }
                        I->add_arc(g_b, u);
                        I->remove_arc(*iter.arc);
                    } else {
                        assert(false);
                    }
                    iter++;
                }

                if (PROFILE) {
                    auto end_step = std::chrono::high_resolution_clock::now();
                    time_step_4 += std::chrono::duration_cast<std::chrono::nanoseconds>(end_step - start_step).count() / 1e6;
                    start_step = std::chrono::high_resolution_clock::now();
                }
                // Step 5. If k = 1, connect f with the -1 face, and connect f
                // with the black subfaces of the grey subfaces of g, otherwise.
                if (k == 1) {
                    NodePtr zero = I->node(I->rank(-1)[0]);
                    I->add_arc(f, zero);
                } else {
                    std::vector<int> V;
                    V.clear();
                    for (int i_u : g->_grey_subfaces) {
                        NodePtr u = I->node(i_u);
                        for (int i_v : u->_black_subfaces) {
                            NodePtr v = I->node(i_v);
                            if (PROFILE) {
                                // TODO
                            }
                            if (v->_black_bit == 0) {
                                V.push_back(v->_id);
                                v->_black_bit = 1;
                            }
                        }
                    }
                    if (PROFILE) {
                        // TODO
                    }
                    for (int i_v : V) {
                        NodePtr v = I->node(i_v);
                        v->_black_bit = 0;
                        I->add_arc(f, v);
                    }
                }

                if (PROFILE) {
                    auto end_step = std::chrono::high_resolution_clock::now();
                    time_step_5 += std::chrono::duration_cast<std::chrono::nanoseconds>(end_step - start_step).count() / 1e6;
                    start_step = std::chrono::high_resolution_clock::now();
                }
                // Step 6. Update the interior points for f, g_a, and g_b.
                f->update_interior_point(eps);
                g_a->update_interior_point(eps);
                g_b->update_interior_point(eps);

                if (PROFILE) {
                    auto end_step = std::chrono::high_resolution_clock::now();
                    time_step_6 += std::chrono::duration_cast<std::chrono::nanoseconds>(end_step - start_step).count() / 1e6;
                    start_step = std::chrono::high_resolution_clock::now();
                }

                break; }
                default: {
                    std::cout << "g: " << get_color_ah_string(g->_color) << std::endl;
                    assert(false);
                }
            }
        }
    }

    for (int k = 0; k < d + 1; k++) {
        for (NodePtr u : L[k]) {
            u->_color = COLOR_AH_WHITE;
            u->_grey_subfaces.clear();
            u->_black_subfaces.clear();
        }
    }

    if (PROFILE) {
        auto end = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < d + 1; k++) {
            std::cout << "  L(" << k << "):" << L[k].size() << std::endl;
        }
        std::cout << "  total: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
        << " ms" << std::endl;
        std::cout << " step 1: " << time_step_1 << " ms" << std::endl;
        std::cout << " step 2: " << time_step_2 << " ms" << std::endl;
        std::cout << " step 3: " << time_step_3 << " ms" << std::endl;
        // std::cout << " copy 3: " << time_step_3_copy << " ms" << std::endl;
        // std::cout << "  ins 3: " << time_step_3_ins << " ms" << std::endl;
        std::cout << " step 4: " << time_step_4 << " ms" << std::endl;
        std::cout << " step 5: " << time_step_5 << " ms" << std::endl;
        std::cout << " step 6: " << time_step_6 << " ms" << std::endl;
        std::cout << "    sum: " << time_step_1 + time_step_2 + time_step_3 + time_step_4 + time_step_5 + time_step_6 << " ms" << std::endl;
    }
    
}