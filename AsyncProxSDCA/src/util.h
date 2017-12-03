#ifndef _UTIL_H_
#define _UTIL_H_

#include <vector>
#include <string>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <iostream>
#include <algorithm>
#include "src/data_type.h"

// read libsvm format data
// https://github.com/cjlin1/libsvm/blob/2fdc614c1526970b4412ba1db0cfcf772023ab61/matlab/libsvmread.c
namespace {

static char *line;
static int max_line_len;

static inline char* readline(FILE *input) {
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL) {
        max_line_len *= 2;
        line = (char *) realloc(line, max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

};

template <typename T> 
bool LoadLibsvm(SparseMatrix<T>& mat, 
    std::vector<T>& y, const std::string& filename) {
    FILE *fp = fopen(filename.c_str(), "r");
    char* endptr = nullptr;
    
    if (fp == nullptr) {
        std::cerr << "can't open input file " << filename << std::endl;
        return false;
    }

    max_line_len = 1024;
    line = (char *)malloc(max_line_len * sizeof(char));

    unsigned int n = 0, d = 0, k = 0;
    T this_val = 0;

    while (readline(fp) != nullptr) {
        char *p = strtok(line, " \t");
        if (p == nullptr) {
            std::cerr << "error: empty line" << std::endl;
            return false;
        }

        // label
        y.push_back(static_cast<T>(strtod(p, &endptr)));

        while (1) {
            char* idx = strtok(nullptr, ":");  // index:value
            char* val = strtok(nullptr, " \t");
            if (val == nullptr) break;

            k = strtol(idx, nullptr, 10) - 1;
            d = std::max(d, k);

            this_val = static_cast<T>(strtod(val, &endptr));
            std::cout << n << ", " << k << ", " << this_val << std::endl;
            mat.append(n, k, this_val);
        }

        if (static_cast<unsigned int>(y.size()) <= (++n)) {
            y.reserve(y.size() * 2);
        }
    }

    fclose(fp);
    free(line);
    ++d;

    return true;
}

//template <typename T> 
//bool LoadLibsvmBinary(SparseMatrix<T>& mat, 
//    std::vector<T>& y, const std::string& filename) {
//
//}
//

template <class rng>
void ShuffleData(SparseMatrix<T>& mat, std::vector<T>& y, rng& rg) {
    using distr_t = std::uniform_int_distribution<size_t>;
    using param_t = distr_t::param_type;

    distr_t D;
    size_t n = y.size(), j;
    for (size_t i = n - 1; i > 0; --i) {
        j = D(rg, param_t(0, i));
        std::swap(mat.indices[i], mat.indices[j]);
        std::swap(mat.values[i], mat.values[j]);
        std::swap(y[i], y[j]);
    }
}

template <class Func>
void ParallelRun(const Func& func, size_t num_thread = 0) {
    if (num_thread == 0) {
        num_thread = std::thread::hardware_concurrency();
    }

    std::vector<std::thread> threads(num_thread);
    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i] = std::threads(func, i);
    }

    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
}

#endif
