#include <fstream>
#include <iostream>
#include <getopt.h>
#include "config.h"

using namespace std;

namespace {
void printUsage(void) {
    std::cout << "Usage: ./train -f train_file_path -m output_model_file_path [options]..." << std::endl
        << "options:" << std::endl
        << "--l1 l1 : l1 regularization parameter (default=1e-3)" << std::endl
        << "--l2 l2 : l2 regularization parameter (default=1e-3)" << std::endl
        << "--gam gamma : smoothed hinge's parameter (default=0.05)" << std::endl
        << "--iter max_iter : max number of iteration for optimization (default=10000)" << std::endl
        << "--tol tol : stopping tolerance for optimization (default=1e-9)" << std::endl
        << "--thread num : number of thread (default=1)" << std::endl
        << "               if = 0, use hardware concureency" << std::endl
        << "--help : print usage" << std::endl;
    std::cout << std::endl << std::endl;
}

void train(
    const string& train_file_path,
    const string& output_model_file_path,
    double l1, double l2, double gam,
    size_t max_iter, double tol, 
    size_t num_thread) {

    Config config;
    config.train_file_path = train_file_path;
    config.output_model_file_path = output_model_file_path;

    config.num_thread = num_thread;
    config.max_iter = max_iter;

    config.algorithm.l1 = l1;
    config.algorithm.l2 = l2;
    config.algorithm.gam = gam;
}
}

int main(int argc, char *argv[])
{
    int opt;
    int opt_idx = 0;

    std::string train_file_path;
    std::string output_model_file_path;

    double l1 = 1e-3;
    double l2 = 1e-3;
    double gam = 0.05;
    size_t max_iter = 10000;
    double tol = 1e-9;
    size_t num_thread = 1;

    static struct option long_options[] = {
        {"l1", required_argument, NULL, 'a'},
        {"l2", required_argument, NULL, 'b'},
        {"gam", required_argument, NULL, 'g'},
        {"iter", required_argument, NULL, 'n'},
        {"tol", required_argument, NULL, 'e'},
        {"thread", required_argument, NULL, 't'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "f:m:", long_options, &opt_idx)) != -1) {
        switch (opt) {
            case 'f':
                train_file_path = optarg;
                break;
            case 'm':
                output_model_file_path = optarg;
                break;
            case 'a':
                l1 = atof(optarg);
                break;
            case 'b':
                l2 = atof(optarg);
                break;
            case 'g':
                gam = atof(optarg);
                break;
            case 'n':
                max_iter = (size_t) atoi(optarg);
                break;
            case 'e':
                tol = atof(optarg);
                break;
            case 't':
                num_thread = (size_t) atoi(optarg);
                break;
            case 'h':
            default:
                printUsage();
                exit(0);
        }
    }

    if (train_file_path.size() == 0 || output_model_file_path.size() == 0) {
        printUsage();
        exit(1);
    }
        
    train(train_file_path, output_model_file_path, l1, l2, gam, max_iter, tol, num_thread);
    return 0;
}
