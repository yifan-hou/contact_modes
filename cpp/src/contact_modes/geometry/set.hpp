#pragma once
#include <vector>


template<class T>
class Set {
public:
    std::vector<T> data;

    Set();

    void clear() {
        data.clear();
    }

    int size() {
        return data.size();
    }

    bool contains(const T& t) {

    }

    void insert(const T& t) {

    }

    bool remove(const T& t) {

    }
};