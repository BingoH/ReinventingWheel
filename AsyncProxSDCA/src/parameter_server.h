#ifndef _PARAMETER_SERVER_H_
#define _PARAMETER_SERVER_H_

#include <string>
#include <unordered_map>
#include <vector>

template <typename T>
class ParameterServer {
public:
    ParameterServer() {}
    virtual ~ParameterServer() {}

    ParameterServer(const ParameterServer&) = delete;
    void operator=(const ParameterServer&) = delete;

    // Add a table to parameter server
    bool AddTable(const std::string& table_name, size_t table_size) {
        if (tables_.find(table_name) != tables_.end()) {
            return false;
        }
        tables_.emplace(table_name, std::vector<T>(table_size));
        return true;
    }

    // Get a table by name
    const std::vector<T>* GetTable(const std::string& table_name) const {
        auto iter = tables_.find(table_name);
        if (iter == tables_.end()) {
            return nullptr;
        } 
        return &(iter->second);
    }

    // Get the parameter from a table by a given index
    bool Get(const std::string& table_name, size_t idx, T* value) const {
        auto iter = tables_.find(table_name);
        if (iter == tables_.end()) {
            return false;
        }
        const auto& table = iter->second;
        if (idx >= table.size()) {
            return false;
        }

        *value = table[idx];
        return true;
    }

    // Get the parameters from a table by a set of indices
    bool Get(const std::string& table_name, const std::vector<size_t>& indices, 
        std::vector<T>* parameters) const {
        auto iter = tables_.find(table_name);
        if (iter == tables_.end()) {
            return false;
        }

        const auto& table = iter->second;
        for (auto idx : indices) {
            if (idx >= table.size()) {
                continue;
            }
            parameters->emplace_back(table[idx]);
        } 
        return true;
    }

    // Set the parameter to a table by a given index/value pair
    bool Set(const std::string& table_name, size_t idx, T value) {
        auto iter = tables_.find(table_name);
        if (iter == tables_.end()) {
            return false;
        }
        auto& table = iter->second;
        if (idx >= table.size()) {
            return false;
        }

        table[idx] = value;
        return true;
    }
    // Set parameters by given indices and values
    bool Set(const std::string& table_name, const std::vector<size_t>& indices,
        const std::vector<T>& parameters) {
        auto iter = tables_.find(table_name);
        if (iter == tables_.end()) {
            return false;
        }
        if (indices.size() != parameters.size()) {
            return false;
        }

        auto& table = iter->second;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= table.size()) {
                continue;
            }
            table[indices[i]] = parameters[i];
        }
        return true;
    }
    // Increase the parameter of a table by a given set of index/value pair
    bool Increase(const std::string& table_name, size_t idx, T value) {
        auto iter = tables_.find(table_name);
        if (iter == tables_.end()) {
            return false;
        }

        auto& table = iter->second;
        if (idx >= table.size()) {
            return false;
        }
        table[idx] += value;
        return true;
    }
    // Increase parameters of a table by given indices and values
    bool Increase(const std::string& table_name, const std::vector<size_t>& indices,
        const std::vector<T>& parameters) {
        auto iter = tables_.find(table_name);
        if (iter == tables_.end()) {
            return false;
        }
        if (indices.size() != parameters.size()) {
            return false;
        }

        auto& table = iter->second;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= table.size()) {
                continue;
            }
            table[indices[i]] += parameters[i];
        }
        return true;
    }

    void Clear() {
        tables_.clear();
    }
private:
    std::unordered_map<std::string, std::vector<T> > tables_;
};

#endif
