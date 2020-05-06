#pragma once
#include <Eigen/Dense>
#include <string>
#include <memory>
#include <contact_modes/geometry/incidence_graph.hpp>


class NodePython;
class IncidenceGraphPython;

typedef std::shared_ptr<NodePython> NodePythonPtr;
typedef std::shared_ptr<IncidenceGraphPython> IncidenceGraphPythonPtr;

class NodePython {
public:
    // On creation, assumes ownership of Node*.
    NodePython(Node* node);

    int rank();
    Eigen::VectorXd interior_point();
    Eigen::VectorXi position();
    std::string     sign_vector();
    std::vector<NodePythonPtr> subfaces();
    std::vector<NodePythonPtr> superfaces();

    // Implementation details.
protected:
    Node* _node;
    IncidenceGraphPythonPtr _graph_python;
};

class IncidenceGraphPython {
public:
    // On creation, assumes ownership of IncidenceGraph*.
    IncidenceGraphPython(IncidenceGraph* graph);

    Eigen::MatrixXd A() const;
    Eigen::VectorXd b() const;
    int dim() const;
    int num_k_faces(int k) const;
    int num_incidences() const;
    int num_nodes() const;
    NodePythonPtr node(int i);
    std::vector<NodePythonPtr>   rank(int k);
    void update_sign_vectors(double eps);
    std::vector<Eigen::VectorXd> interior_points();
    std::vector<Eigen::VectorXd> interior_points(int k);
    std::vector<Eigen::VectorXi> positions();
    std::vector<Eigen::VectorXi> positions(int k);
    std::vector<std::string>     sign_vectors();
    std::vector<std::string>     sign_vectors(int k);

protected:
    IncidenceGraph* _graph;
};