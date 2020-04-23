#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <unordered_set>
#include <string>
#include <boost/pool/object_pool.hpp>


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

enum {
    STRICTLY_LESS       = 1,
    STRICTLY_GREATER    = 2,
    EQUAL               = 4,
    LESS_THAN_EQUAL     = 5,
    GREATER_THAN_EQUAL  = 6,
    INCOMPARABLE        = 8
};

int partial_order(const std::string& lhs, const std::string& rhs);

struct Arc {
    Node* dst;
    Arc*  _dst_arc;
    Arc*  _next;
    Arc*  _prev;

    Arc();

    friend std::ostream& operator<<(std::ostream& out, const Arc& arc);
};

class ArcListIterator;

typedef std::vector<Arc> Arcs;

class ArcList {
public:
    Arc* _begin;
    int  _size;

    ArcList();

    int size() { return _size; }
    ArcListIterator begin();
    ArcListIterator end();

    friend IncidenceGraph;

    void _add_arc(Arc* arc);
    void _remove_arc(Arc* arc);
};

class ArcListIterator {
public:
    using value_type = int;
    using reference = int;
    using iterator_category = std::input_iterator_tag;
    using pointer = int*;
    using difference_type = void;

    Arc*     arc;
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
    Node*  operator*()  const;
    Node** operator->() const;

    friend bool operator==(const ArcListIterator& lhs, const ArcListIterator& rhs);
    friend bool operator!=(const ArcListIterator& lhs, const ArcListIterator& rhs);
};

class Node {
public:
    int8_t              rank;
    int8_t              _color;
    int8_t              _black_bit;
    int8_t              _sign_bit;
    int                 _sign_bit_n;
    ArcList             superfaces;
    ArcList             subfaces;
    IncidenceGraph*     _graph;
    std::vector<Node*>  _grey_subfaces;
    std::vector<Node*>  _black_subfaces;
    std::string         _key; // target sign vector

    int                 _id;
    Eigen::VectorXd     interior_point;
    Position            position;
    SignVector          sign_vector;

    Node(int k);
    void reset();

    IncidenceGraph* graph() { return _graph; }

    void update_interior_point(double eps);
    void update_position(double eps);
    void update_sign_vector(double eps);

    friend std::ostream& operator<<(std::ostream& out, Node& node);
};

typedef std::vector<Node*> Rank;

struct aligned_allocator_mm_malloc_free
{
  typedef std::size_t size_type; //!< An unsigned integral type that can represent the size of the largest object to be allocated.
  typedef std::ptrdiff_t difference_type; //!< A signed integral type that can represent the difference of any two pointers.

  static char * malloc BOOST_PREVENT_MACRO_SUBSTITUTION(const size_type bytes)
  { return static_cast<char *>((_mm_malloc)(bytes, 64)); }
  static void free BOOST_PREVENT_MACRO_SUBSTITUTION(char * const block)
  { (_mm_free)(block); }
};

typedef boost::object_pool<Arc, aligned_allocator_mm_malloc_free> ArcPool;
typedef boost::object_pool<Node, aligned_allocator_mm_malloc_free> NodePool;

class IncidenceGraph : public std::enable_shared_from_this<IncidenceGraph> {
public:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<Node*> _nodes;
    std::vector<Rank>  _lattice;
    int                _num_nodes_created;
    int                _num_arcs_created;
    NodePool           _node_pool;
    ArcPool            _arc_pool;

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

    Node*       node(int id) { return _nodes[id]; }
    Node*  make_node(int k);
    void    add_node_to_rank(Node* node);
    void remove_node_from_rank(Node* node);

    // O(1) add
    void add_arc(Node* sub, Node* super,
                 Arc* arc1=nullptr, Arc* arc2=nullptr);   
    // O(1) remove
    void remove_arc(Arc* arc);

    Rank& rank(int k);

    void print_neighbor_stats();
};