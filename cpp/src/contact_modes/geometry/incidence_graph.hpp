#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <list>
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

class Arc {
public:
    int dst_id;
    int src_id; // TODO unnecessary
    int16_t _dst_arc_idx;
    int16_t _src_arc_idx;
    int16_t _next;
    int16_t _prev;

    Arc();

    friend std::ostream& operator<<(std::ostream& out, const Arc& arc);
};

class ArcListIterator;

class ArcList {
public:
    int16_t          _begin;
    int16_t          _size;
    int16_t          _empty_size;
    std::vector<Arc> arcs;
    // std::list<int>   _empty_indices;
    std::vector<int> _empty_indices;

    ArcList();

    int size() { return _size; }
    ArcListIterator begin();
    ArcListIterator end();

    friend IncidenceGraph;

protected:
    int _next_empty_index();
    void _add_arc(Arc& arc, int index);
    void _remove_arc(const Arc& arc, NodePtr& src);
};

class ArcListIterator {
public:
    using value_type = int;
    using reference = int;
    using iterator_category = std::input_iterator_tag;
    using pointer = int*;
    using difference_type = void;

    Arc* arc;
    ArcList* arc_list;

    class postinc_return {
    public:
        int value;
        postinc_return(int value_) { value = value_; }
        int operator*() { return value; }
    };

    ArcListIterator(ArcList* arc_list, Arc* arc);

    ArcListIterator& operator++();
    ArcListIterator  operator++(int);
    // bool operator==(const ArcListIterator& other) const;
    // bool operator!=(const ArcListIterator& other) const;
    int  operator*() const;
    int* operator->() const;

    friend bool operator==(const ArcListIterator& lhs, const ArcListIterator& rhs);
    friend bool operator!=(const ArcListIterator& lhs, const ArcListIterator& rhs);
};

class Node {
public:
    int                 _id;
    int8_t              rank;
    int8_t              _color;
    int8_t              _black_bit;
    int8_t              _sign_bit;
    int                 _sign_bit_n;
    std::vector<int>    _grey_subfaces;
    std::vector<int>    _black_subfaces;
    std::string         _key; // target sign vector
    IncidenceGraphPtr   _graph;
    ArcList             superfaces;
    ArcList             subfaces;
    Eigen::VectorXd     interior_point;
    Position            position;
    SignVector          sign_vector;

    Node(int k);

    IncidenceGraphPtr graph() { return _graph; }

    void update_interior_point(double eps);
    void update_position(double eps);
    void update_sign_vector(double eps);
};

typedef std::vector<int> Rank;

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
    void     add_node_to_rank(NodePtr node);
    void  remove_node(NodePtr node);

    void add_arc(NodePtr& src, NodePtr& dst);   // O(1) add
    void remove_arc(const Arc& arc);            // O(1) remove

    Rank& rank(int k);
};