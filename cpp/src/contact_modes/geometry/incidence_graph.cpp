#include <contact_modes/geometry/incidence_graph.hpp>
#include <contact_modes/geometry/linear_algebra.hpp>
#include <iostream>

static int DEBUG=0;

std::string get_color_ah_string(int color) {
    switch (color) {
        case COLOR_AH_WHITE:    return "WHITE";
        case COLOR_AH_PINK:     return "PINK";
        case COLOR_AH_RED:      return "RED";
        case COLOR_AH_CRIMSON:  return "CRIMSON";
        case COLOR_AH_GREY:     return "GREY";
        case COLOR_AH_BLACK:    return "BLACK";
        case COLOR_AH_GREEN:    return "GREEN";
        default:                return "INVALID";
    }
}

int get_position(double v, double eps) {
    assert(eps > 0);
    if (v > eps) {
        return 1;
    } else if (v < -eps) {
        return -1;
    } else {
        return 0;
    }
}

char get_sign(double v, double eps) {
    assert(eps > 0);
    if (v > eps) {
        return '+';
    } else if (v < -eps) {
        return '-';
    } else {
        return '0';
    }
}

void get_position(const Eigen::VectorXd& v, Eigen::VectorXi& pos, double eps) {
    eps = abs(eps);
    pos.resize(v.size());
    for (int i = 0; i < v.size(); i++) {
        pos[i] = get_position(v[i], eps);
    }
}

Eigen::VectorXi get_position(const Eigen::VectorXd& v, double eps) {
    Eigen::VectorXi position;
    get_position(v, position, eps);
    return position;
}

void get_sign_vector(const Eigen::VectorXd& v, std::string& sv, double eps) {
    eps = abs(eps);
    sv.resize(v.size());
    for (int i = 0; i < v.size(); i++) {
        sv[i] = get_sign(v[i], eps);
    }
}

std::string get_sign_vector(const Eigen::VectorXd& v, double eps) {
    std::string sv;
    get_sign_vector(v, sv, eps);
    return sv;
}

void arg_where(const std::string& sv, char s, Eigen::VectorXi& idx) {
    int cnt = 0;
    for (int i = 0; i < sv.size(); i++) {
        cnt += sv[i] == s;
    }
    idx.resize(cnt);
    int k = 0;
    for (int i = 0; i < sv.size(); i++) {
        if (sv[i] == s) {
            idx[k++] = i;
        }
    }
}

void arg_equal(const std::string& a, 
               const std::string& b, 
               Eigen::VectorXi& idx) 
{
    assert(a.size() == b.size());
    int cnt = 0;
    for (int i = 0; i < a.size(); i++) {
        cnt += a[i] == b[i];
    }
    idx.resize(cnt);
    int k = 0;
    for (int i = 0; i < a.size(); i++) {
        if (a[i] == b[i]) {
            idx[k++] = i;
        }
    }
}

void arg_not_equal(const std::string& a, 
                   const std::string& b, 
                   Eigen::VectorXi& idx)
{
    assert(a.size() == b.size());
    int cnt = 0;
    for (int i = 0; i < a.size(); i++) {
        cnt += a[i] != b[i];
    }
    idx.resize(cnt);
    int k = 0;
    for (int i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            idx[k++] = i;
        }
    }
}

Node::Node(int k) {
    this->rank = k;
    this->interior_point.resize(0);
    this->position.resize(0);
    this->sign_vector.clear();
    this->subfaces.clear();
    this->superfaces.clear();
    this->_id = 0;
    this->_color = COLOR_AH_WHITE;
    this->_grey_subfaces.clear();
    this->_black_subfaces.clear();
    this->_key.clear();
    this->_black_bit = false;
    this->_sign_bit_n = 0;
    this->_sign_bit = 0;
    this->_graph = nullptr;
}

void Node::update_interior_point(double eps) {
    if (this->rank == -1 || this->rank == this->graph()->dim() + 1) {
        return;
    }

    // Case: Vertex
    else if (this->rank == 0) {
        // Solve linear equations for unique intersection point.
        Eigen::VectorXi idx;
        arg_where(this->_key, '0', idx);
        if (DEBUG) {
            assert(idx.size() == _graph->A.cols());
        }
        Eigen::MatrixXd A0;
        Eigen::VectorXd b0;
        get_rows(this->graph()->A, idx, A0);
        get_rows(this->graph()->b, idx, b0);
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;
        qr.setThreshold(eps);
        qr.compute(A0);
        this->interior_point = qr.solve(b0);
    } 

    // Case: Edge
    else if (this->rank == 1) {

        // Case: 2 vertices: Average interior points of vertices.
        if (this->subfaces.size() == 2) {
            auto iter = this->subfaces.begin();
            int i0 = *iter++;
            int i1 = *iter;
            NodePtr v0 = this->graph()->node(i0);
            NodePtr v1 = this->graph()->node(i1);
            this->interior_point = (v0->interior_point + v1->interior_point) / 2.0;
        }

        // Case: 1 vertex: Pick an interior point along the unbounded edge
        // (ray).
        else if (this->subfaces.size() == 1) {
            // Get vertex.
            NodePtr v = this->graph()->node(*this->subfaces.begin());
            // Get edge key up to vertex key length.
            std::string e_sv = this->_key.substr(0, v->_key.size());
            if (DEBUG) {
                assert(v->_key.size() <= this->_key.size());
            }
            // Compute edge direction.
            Eigen::VectorXi id0;
            arg_where(this->_key, '0', id0);
            Eigen::MatrixXd A0 = get_rows(this->graph()->A, id0);
            Eigen::MatrixXd kernel = kernel_basis(A0, eps);
            if (DEBUG) {
                assert(kernel.cols() == 1);
            }
            // Get vertex key.
            std::string v_sv = v->_key;
            if (DEBUG) {
                // std::cout << e_sv << std::endl;
                // std::cout << v_sv << std::endl;
            }
            // Find an index where edge and vertex keys disagree.
            Eigen::VectorXi idx;
            arg_not_equal(e_sv, v_sv, idx);
            int i = idx[0];
            if (DEBUG) {
                assert(e_sv[i] != '0');
                assert(v_sv[i] == '0');
            }
            // Determine ray direction.
            Eigen::MatrixXd A_i = this->graph()->A.row(i);
            Eigen::VectorXd b_i = this->graph()->b.row(i);
            double dot = (A_i * (v->interior_point + kernel) - b_i).sum();
            double sgn = e_sv[i] == '+' ? 1 : -1;
            if (dot * sgn > 0) {
                this->interior_point = v->interior_point + kernel;
            } else {
                this->interior_point = v->interior_point - kernel;
            }
        } else {
            assert(false);
        }
    }

    // Case: k-face with k >= 2: Average interior points of vertices.
    else {
        this->interior_point.resize(this->graph()->A.cols());
        this->interior_point.setZero();
        auto iter = this->subfaces.begin();
        auto  end = this->subfaces.end();
        while (iter != end) {
            int idx = *iter++;
            NodePtr f = _graph->node(idx);
            this->interior_point += f->interior_point;
        }
        this->interior_point /= this->subfaces.size();
    }
    if (DEBUG) {
        // Assert sign vector matches key.
        this->update_sign_vector(eps);
        bool match = this->_key == this->sign_vector.substr(0, this->_key.size());
        if (!match) {
            std::cout << "rank: " << this->rank << "\n"
                      << " key: " << this->_key << "\n"
                      << "  sv: " << this->sign_vector << std::endl;
        }
        assert(match);
    }
}

void Node::update_position(double eps) {
    if (this->rank == -1 || this->rank == this->_graph->dim() + 1) {
        return;
    } else {
        Eigen::VectorXd res = _graph->A * this->interior_point - _graph->b;
        get_position(res, this->position, eps);
    }
    // if (DEBUG) {
    //     std::cout << "rank: " << this->rank << std::endl;
    //     std::cout << _graph->A << std::endl;
    //     std::cout << _graph->b << std::endl;
    //     std::cout << "int pt\n" << this->interior_point << std::endl;
    // }
}

void Node::update_sign_vector(double eps) {
    update_position(eps);
    sign_vector.clear();
    for (int i = 0; i < this->position.size(); i++) {
        if (this->position[i] == 0) {
            sign_vector.push_back('0');
        } else if (this->position[i] == 1) {
            sign_vector.push_back('+');
        } else if (this->position[i] == -1) {
            sign_vector.push_back('-');
        } else {
            assert(false);
        }
    }
}

IncidenceGraph::IncidenceGraph(int d) 
    : _num_nodes_created(0)
{
    this->_lattice.resize(d + 3);
    for (int i = 0; i < d + 3; i++) {
        this->_lattice[i].clear();
    }
}

IncidenceGraph::~IncidenceGraph() {
}

int IncidenceGraph::dim() {
    return this->A.cols();
}

int IncidenceGraph::num_incidences() {
    return 0;
}

void IncidenceGraph::add_hyperplane(const Eigen::VectorXd& a, double d) {
    int n = this->A.rows();
    int m = this->A.cols();
    Eigen::MatrixXd A(n + 1, m);
    A << this->A, a.transpose();
    Eigen::VectorXd b(n + 1);
    b << this->b, d;
    this->A = A;
    this->b = b;
    // if (DEBUG) {
    //     std::cout << "a\n" << a << std::endl;
    //     std::cout << "A\n" << this->A << std::endl;
    //     std::cout << "b\n" << d << std::endl;
    //     std::cout << "b\n" << this->b << std::endl;
    // }
}

void IncidenceGraph::update_positions(double eps) {
    for (int i = 0; i < this->_nodes.size(); i++) {
        this->_nodes[i]->update_position(eps);
    }
}

void IncidenceGraph::update_sign_vectors(double eps) {
    for (int i = 0; i < this->_nodes.size(); i++) {
        this->_nodes[i]->update_sign_vector(eps);
    }
}

Positions IncidenceGraph::get_positions() {
    Positions P;
    for (int i = 0; i < this->_nodes.size(); i++) {
        if (this->_nodes[i]->position.size() != 0) {
            P.push_back(this->_nodes[i]->position);
        }
    }
    return P;
}

SignVectors IncidenceGraph::get_sign_vectors() {
    SignVectors S;
    for (int i = 0; i < this->_nodes.size(); i++) {
        if (!this->_nodes[i]->sign_vector.empty()) {
            S.push_back(this->_nodes[i]->sign_vector);
        }
    }
    return S;
}

NodePtr IncidenceGraph::make_node(int k) {
    NodePtr node = std::make_shared<Node>(k);
    node->_id = this->_nodes.size();
    node->_graph = shared_from_this();
    this->_nodes.push_back(node);
    return node;
}

void IncidenceGraph::add_node(NodePtr node) {
    this->rank(node->rank).insert({node->_key, node->_id});
}

void IncidenceGraph::remove_node(NodePtr node) {
    // Remove arcs.
    for (int f_id : node->subfaces) {
        this->_nodes[f_id]->superfaces.erase(node->_id);
    }
    for (int g_id : node->superfaces) {
        NodePtr& g = this->_nodes[g_id];
        int r = g->subfaces.erase(node->_id);
        // Check that we never have to remove g/f from its parents grey or black
        // subface vectors. Only relevant during a call to increment_arrangement
        if (DEBUG) {
            if (r > 0) {
                if (node->_color == COLOR_AH_GREY) {
                    const std::vector<int>& grey = g->_grey_subfaces;
                    int c = std::count(grey.begin(), grey.end(), node->_id);
                    assert(c == 0);
                }
                else if (node->_color == COLOR_AH_BLACK) {
                    const std::vector<int>& black = g->_black_subfaces;
                    int c = std::count(black.begin(), black.end(), node->_id);
                    assert(c == 0);
                }
            }
        }
    }
    // Remove node.
    this->rank(node->rank).erase(node->_key);
}

NodePtr IncidenceGraph::get_node(const std::string& key, int k) {
    auto iter = rank(k).find(key);
    if (iter == rank(k).end()) {
        return nullptr;
    } else {
        return this->_nodes[iter->second];
    }
}

Rank& IncidenceGraph::rank(int k) {
    return this->_lattice[k+1];
}