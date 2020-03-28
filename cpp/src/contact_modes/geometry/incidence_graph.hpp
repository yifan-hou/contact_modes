#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <string>


class Node;
typedef std::shared_ptr<Node> NodePtr;

class IncidenceGraph;
typedef std::shared_ptr<IncidenceGraph> IncidenceGraphPtr;

enum {
    COLOR_AH_WHITE   = 0,   // if cl(f) ∩ h = ∅
    COLOR_AH_PINK    = 1,   // if cl(f) ∩ h ≠ ∅, f ∩ h = ∅
    COLOR_AH_RED     = 2,   // if f ∩ h ≠ ∅, f ⊈ h
    COLOR_AH_CRIMSON = 3,   // if f ⊆ h
    COLOR_AH_GREY    = 4,   // if cl(f) ∩ h ≠ ∅, f ∩ h = ∅
    COLOR_AH_BLACK   = 5,   // if f ⊆ h
    COLOR_AH_GREEN   = 6,   // if some subface is non-white
};

std::string get_color_ah_string(int color);

typedef Eigen::VectorXi Position;
typedef std::vector<Position> Positions;
typedef std::string SignVector;
typedef std::vector<SignVector> SignVectors;

int  get_position(double x, double eps);
char get_sign(double x, double eps);
void get_position(const Eigen::VectorXd& v, Eigen::VectorXi& pos, double eps);
Eigen::VectorXi get_position(const Eigen::VectorXd& v, double eps);
void get_sign_vector(const Eigen::VectorXd& v, std::string& sv, double eps);
std::string get_sign_vector(const Eigen::VectorXd& v, double eps);
void arg_where(const std::string& sv, char s, Eigen::VectorXi& idx);
void arg_equal(const std::string& a, const std::string& b, Eigen::VectorXi& idx);
void arg_not_equal(const std::string& a, const std::string& b, Eigen::VectorXi& idx);
// bool less_than(const Eigen::VectorXi& a, const Eigen::VectorXi& b);
// bool less_than(const std::string& a, const std::string& b);

class Node {
public:
    int                 rank;
    Eigen::VectorXd     interior_point;
    Position            position;
    SignVector          sign_vector;
    std::set<int>       superfaces;
    std::set<int>       subfaces;
    int                 _id;
    int                 _color;
    // std::set<int>       _grey_subfaces;
    // std::set<int>       _black_subfaces;
    std::vector<int>    _grey_subfaces;
    std::vector<int>    _black_subfaces;
    std::string         _key;
    bool                _black_bit;
    int                 _sign_bit_n;
    int                 _sign_bit;
    IncidenceGraphPtr   _graph;

    Node(int k);

    IncidenceGraphPtr graph() { return _graph; }

    void update_interior_point(double eps);
    void update_position(double eps);
    void update_sign_vector(double eps);
};

typedef std::map<std::string, int> Rank;

class IncidenceGraph : public std::enable_shared_from_this<IncidenceGraph> {
public:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<NodePtr> _nodes;
    std::vector<Rank>    _lattice;
    int                  _num_nodes_created;

    IncidenceGraph(int d);
    ~IncidenceGraph();

    void set_hyperplanes(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    void add_hyperplane (const Eigen::VectorXd& a, double b);

    int dim();
    int num_k_faces();
    int num_incidences();

    void update_positions(double eps);
    Positions get_positions();

    void update_sign_vectors(double eps);
    SignVectors get_sign_vectors();

    NodePtr      node(int id) { return _nodes[id]; }
    NodePtr make_node(int k);
    void     add_node(NodePtr node);
    void  remove_node(NodePtr node);
    NodePtr  get_node(const std::string& key, int k);

    Rank& rank(int k);
};