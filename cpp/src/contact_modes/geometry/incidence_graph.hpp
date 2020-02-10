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
    // 
    int rank;
    Eigen::VectorXd interior_point;
    std::set<NodePtr> superfaces;
    std::set<NodePtr> subfaces;
    // 
    int _color;
    std::set<NodePtr> _grey_subfaces;
    std::set<NodePtr> _black_subfaces;
    std::string _sv_key;
    int _black_bit;

    Node(int k);
};

class IncidenceGraph {
public:
    std::vector<std::map<std::string, NodePtr> > _lattice;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    IncidenceGraph(int d);
    ~IncidenceGraph();

    int dim();
    int num_incidences();

    void add_halfspace(const Eigen::VectorXd& a, double b);

    void add_node(const NodePtr& node);
    void remove_node(const NodePtr& node);
    NodePtr get_node(const std::string& sv_key, int k);

    std::map<std::string, NodePtr>& rank(int k);
};

typedef std::shared_ptr<IncidenceGraph> IncidenceGraphPtr;