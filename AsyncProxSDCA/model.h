#ifndef _MODEL_H_
#define _MODEL_H_

template <typename Algorithm>
class Model {
public:
    Model() {}
    virtual ~Model() {}

    void Initialize(const Config<Algorithm>& config) {
        config_ = config;
    }

private:
    Config<Algorithm> config_;
    ParameterServer<DataType> parameter_server_;
    std::vector<DataType> model_;
};

#endif
