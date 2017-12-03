#ifndef _MODEL_H_
#define _MODEL_H_

#include <vector>
#include <random>
#include <functional>
#include "src/data_type.h"
#include "src/util.h"

template <typename Algorithm>
class Model {
public:
    using DataType = Algorithm::DataType;
    using WokerType = Algorithm::WorkerType;

    Model() {}
    virtual ~Model() {}

    void Initialize(const Config<Algorithm>& config) {
        config_ = config;

        // create tables in parameter server
        for (const auto& table : config.algorithm.tables) {
            if (!parameter_server_.AddTable(table.first, table.second.length)) {
                std::cout << "Create table " << table.first << " with length "
                    << table.second.length << " failed." << std::endl;
                return;
            }
        }
    }

    void Train() { 
        size_t num_thread;
        if (config_.num_thread == 0) {
            num_thread = std::thread::hardware_concurrency();
        } else {
            num_thread = config_.num_thread;
        }

        std::vector<WorkerType> workers(num_thread);
        for (size_t i = 0; i < num_thread; ++i) {
            workers[i].SetConfig(&config_).SetParameterServer(&parameter_server_);
        }

        // TODO: drop the unaligned tail currently
        size_t n_sam_per_thread = X_train.rows() / num_thread;
        for (size_t i = 0; i < num_thread; ++i) {
            workers[i].SetData(X_train.indices.begin(), X_train.values.begin(),
                i * n_sam_per_thread, n_sam_per_thread);
        }
       
        auto worker_func = [&] (size_t i) {
            workers[i].Train();
        };
        
        ParallelRun(worker_func, num_thread);
    }

    void Save() {
        WorkerType worker;
        worker.SetParameterServer(&parameter_server_);
        worker.SetConfig(&config_);
        worker.SaveModel();
    }

    void Test() {
        // TODO
    }
private:
    Config<Algorithm> config_;
    ParameterServer<DataType> parameter_server_;
};

#endif
