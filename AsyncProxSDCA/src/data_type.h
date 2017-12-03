#ifndef _DATA_TYPE_
#define _DATA_TYPE_

#include <iostream>
#include <vector>
#include <algorithm>

// simple rowmajor sparse matrix 
template<typename T>
class SparseMatrix {
public:
    SparseMatrix() : n_(0), d_(0) {}
    virtual ~SparseMatrix() {}

    // TODO: can only insert to last or next row currently; maybe use other sparse storage
    bool append(size_t ir, size_t jc, const T value) {
        if (ir == n_ - 1) {
            indices[ir].push_back(jc);
            values[ir].push_back(value);
        } else if (ir == n_){
            indices.emplace_back(std::vector<size_t>());
            indices[ir].push_back(jc);
            values.emplace_back(std::vector<T>());
            values[ir].push_back(value);
            //std::cout << "ir: " << ir << "jc: " << jc << std::endl;
        } else {
            return false;
        }
        n_ = std::max(n_, ir + 1);
        d_ = std::max(d_, jc + 1);
        return true;
    }

    size_t rows(void) const {
        return n_;
    }

    size_t cols(void) const {
        return d_;
    }

    void show(void) const {
        std::cout << "n: " << n_ << ", d: " << d_ << std::endl;
        for (size_t i = 0; i < n_; ++i) {
            auto it1 = indices[i].begin();
            auto it2 = values[i].begin();
            for (; it1 != indices[i].end(); ++it1, ++it2) {
                std::cout << (*it1) << " : " << (*it2) << "\t";
            }
            std::cout << std::endl;
        }
    }

    size_t n_, d_;
    // Simply store nnz indices and values per row
    std::vector<std::vector<size_t> > indices;
    std::vector<std::vector<T> > values;
};

#endif
