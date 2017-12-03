#ifndef _PSDCA_H_
#define _PSDCA_H_

#include <vector>
#include <unordered_map>
#include <random>
#include "src/parameter_server.h"
#include "src/worker.h"
#include "src/config.h"

class ProxSdcaWorker;

class ProxSdca {
public:
    using WorkerType = ProxSdcaWorker;
    // consider double precision
    using DataType = double;

    DataType l1;
    DataType l2;
    DataType gam;
    DataType tol;

    size_t num_all_sample;

    struct TableConfig {
        size_t length;
    };

    std::unordered_map<std::string, TableConfig> tables;
};

using dtype = ProxSdca::DataType;
class ProxSdcaWorker : public Worker<dtype> {
public:
    ProxSdcaWorker() {}
    virtual ~ProxSdcaWorker() {}

    ProxSdcaWorker& SetConfig(Config<ProxSdca>* config) {
        config_ = config;
        return *this;
    }

    ProxSdcaWorker& SetData(std::vector<std::vector<size_t> >::iterator idx_begin_it,
        std::vector<std::vector<dtype> >::iterator val_begin_it, 
        size_t offset, size_t num_samples) {
        idx_it_ = std::next(idx_begin_it, offset);
        val_it_ = std::next(val_begin_it, offset);
        offset_ = offset;
        num_samples_ = num_samples;
        cache_.resize(num_samples);
        return *this;
    }

    virtual bool Train() {
        // compute sample norm
        for (size_t i = 0; i < num_samples_; ++i) {
            dtype sq = 0;
            auto sam_val_it = std::next(val_it_, i);
            for (auto val : *sam_val_it) {
                sq += val * val;
            }
            cache_[i] = sq;
        }

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<size_t> ud(0, num_samples_ - 1);

        const dtype inv_nl2 =  1. / (config_->algorithm.num_all_sample * config_->algorithm.l2);

        // TODO: stop criterion checking
        for (;;) {
            size_t i = ud(mt);
            auto sam_idx_it = std::next(idx_it_, i);
            auto sam_val_it = std::next(val_it_, i);

            size_t size = sam_idx_it->size();
            dtype ptheta;
            Worker<dtype>::GetParameterServer()->Get("theta", offset_ + i, &ptheta);

            dtype xiw = 0;
            std::vector<dtype> weights;
            if (!Worker<dtype>::GetParameterServer()->Get("W", *sam_idx_it, &weights)) {
                return false;
            }
            for (size_t k = 0; k < size; ++k) {
                xiw += weights[k] * sam_val_it->at(k);
            }

            dtype dtheta = (1 - xiw - config_->algorithm.gam * ptheta) / 
                (config_->algorithm.gam + cache_[i] * inv_nl2);
            dtheta = std::max(-ptheta, std::min(1 - ptheta, dtheta));
            // update theta
            Worker<dtype>::GetParameterServer()->Increase("theta", i + offset, dtheta);

            dtype delta = dtheta / static_cast<dtype>(config_->algorithm.num_all_sample);
            dtype XTtheta_i;
            dtype wi;
            for (size_t k = 0; k < size; ++k) {
                size_t idx = sam_idx_it->at(k);
                Worker<dtype>::GetParameterServer()->Get("XTtheta",
                    idx, &XTtheta_i);
                XTtheta_i += delta * sam_val_it->at(k);
                Worker<dtype>::GetParameterServer()->Set("XTtheta", idx, XTtheta_i);
                if (XTtheta_i > config_->algorithm.l1) {
                    wi = XTtheta_i - config_->algorithm.l1;
                } else if (XTtheta_i < -config_->algorithm.l1) {
                    wi = XTtheta_i + config_->algorithm.l1;
                } else {
                    wi = 0;
                }
                wi /= config_->algorithm.l2;
                Worker<dtype>::GetParameterServer()->Set("W", idx, wi);
            }
        }
    }

    Config<ProxSdca>* config_;  // not owned
    // TODO: maybe use ptr
    std::vector<std::vector<size_t> >::iterator idx_it_;  // not owned
    std::vector<std::vector<dtype> >::iterator val_it_;  // not owned
    std::vector<dtype> cache_;  // |x_i|^2
    size_t offset_;  // block offset
    size_t num_samples_;  // block size
};

#endif
