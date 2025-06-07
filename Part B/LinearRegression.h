#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <string>

class LinearRegression {
private:
    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> trainSet, testSet;
    std::vector<double> weights;
    std::vector<double> trainRMSEs;
    std::vector<double> testRMSEs;
    double bias;

public:
    LinearRegression();
    void addData(const std::vector<double>& sample);
    int getLength() const;
    void splitData(double trainRatio = 0.8);
    double computeRMSE(const std::vector<std::vector<double>>& dataset);
    double predict(const std::vector<double>& sample);
    void train(int epochs = 100, double learning_rate = 1e-3, int batch_size = 5);
    const std::vector<std::pair<double, double>> infer();
    const std::vector<double>& getWeights() const;
    const std::vector<double>& getTrainRMSEs() const;
    const std::vector<double>& getTestRMSEs() const;
};  

void readData(LinearRegression& regression, const std::string& filename);
void exportRMSEToCSV(const std::string& filename,
                     const std::vector<double>& trainRMSEs,
                     const std::vector<double>& testRMSEs);

#endif