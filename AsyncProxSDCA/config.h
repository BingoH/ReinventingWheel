#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <string>

template <typename Algorithm>
class Config {
public:
    // for training
    std::string train_file_path;
    std::string output_model_file_path;

    size_t num_thread;
    size_t max_iter;

    Algorithm algorithm;
    
};

#endif
