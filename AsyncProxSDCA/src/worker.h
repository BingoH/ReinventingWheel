#ifndef _WORKER_H_
#define _WORKER_H_

template <typename T>
class Worker {
public:
    Worker() : parameter_server_(nullptr) {}
    virtual ~Worker() {}

    virtual ParameterServer<T>* GetParameterServer() const {
        return parameter_server_;
    }

    virtual Worker& SetParameterServer(ParameterServer<T>* ps) {
        parameter_server_ = ps;
        return *this;
    }

protected:
    ParameterServer<T>* parameter_server_;  // Not owned
};

#endif
