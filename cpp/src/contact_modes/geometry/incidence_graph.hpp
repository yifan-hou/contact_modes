#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>


class Node;
typedef std::shared_ptr<Node> NodePtr;

enum {
    COLOR_AH_WHITE   = 0,
    COLOR_AH_PINK    = 1,
    COLOR_AH_RED     = 2,
    COLOR_AH_CRIMSON = 3,
    COLOR_AH_GREY    = 4,
    COLOR_AH_BLACK   = 5,
    COLOR_AH_GREEN   = 6,
};

class Node {
public:
    int             rank;
    Eigen::VectorXd interior_point;
    Eigen::VectorXi position;
    std::set<int>   superfaces;
    std::set<int>   subfaces;
    int             _id;
    int             _color;
    std::set<int>   _grey_subfaces;
    std::set<int>   _black_subfaces;
    std::string     _key;
    bool            _black_bit;
    bool            _sign_bit_n;
    bool            _sign_bit;
    Node(int k);
};

typedef std::map<std::string, int> Rank;

class IncidenceGraph {
public:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<NodePtr> _nodes;
    std::vector<Rank>    _lattice;

    IncidenceGraph(int d);
    ~IncidenceGraph();

    void set_hyperplanes(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    void add_hyperplane (const Eigen::VectorXd& a, double b);

    int dim();
    int num_k_faces();
    int num_incidences();

    void update_positions();

    void    add_node(const NodePtr& node);
    void remove_node(const NodePtr& node);
    NodePtr get_node(const std::string& key, int k);

    std::map<std::string, NodePtr>& rank(int k);
};

typedef std::shared_ptr<IncidenceGraph> IncidenceGraphPtr;