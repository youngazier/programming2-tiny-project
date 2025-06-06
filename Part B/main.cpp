#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>

#include "../Part A/Vector.cpp"
#include "../Part A/Matrix.cpp"
#include "../Part A/LinearSystem.cpp"

// Compute RMSE between predictions and actual labels
double computeRMSE(const Matrix& X, const Vector& y, const Vector& weights) {
    Vector predictions = X * weights;
    double sumSq = 0.0;
    for (int i = 0; i < y.size(); ++i) {
        double err = predictions[i] - y[i];
        sumSq += err * err;
        if (std::isnan(err) || std::isinf(err)) return std::nan("1");
    }
    return std::sqrt(sumSq / y.size());
}

// Normalize features using z-score
void normalizeFeatures(std::vector<Vector>& features) {
    int total = features.size();
    int dim = features[0].size();
    std::vector<double> mean(dim, 0.0), stddev(dim, 0.0);

    for (const auto& vec : features)
        for (int j = 0; j < dim; ++j)
            mean[j] += vec[j];
    for (int j = 0; j < dim; ++j) mean[j] /= total;

    for (const auto& vec : features)
        for (int j = 0; j < dim; ++j)
            stddev[j] += std::pow(vec[j] - mean[j], 2);
    for (int j = 0; j < dim; ++j)
        stddev[j] = std::sqrt(stddev[j] / total);

    for (auto& vec : features)
        for (int j = 0; j < dim; ++j)
            vec[j] = (vec[j] - mean[j]) / std::max(stddev[j], 1e-12);
}

// Gradient descent training for linear regression with CSV logging
Vector gradientDescent(const Matrix& A_train, const Vector& b_train,
                       const Matrix& A_test, const Vector& b_test,
                       int epochs = 5000, double lr = 1e-4) {
    int n = A_train.numCols();
    Vector weights(n);

    std::ofstream logFile("metrics.csv");
    logFile << "epoch,train_rmse,test_rmse\n";

    std::cout << "Starting training....\n";

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        Vector predictions = A_train * weights;
        Vector gradient(n);
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int i = 0; i < A_train.numRows(); ++i) {
                sum += (predictions[i] - b_train[i]) * A_train(i + 1, j + 1);
            }
            gradient[j] = sum * 2.0 / A_train.numRows();
        }

        for (int j = 0; j < n; ++j) {
            weights[j] -= lr * gradient[j];
        }

        if (epoch % 100 == 0 || epoch == epochs) {
            double rmse_train = computeRMSE(A_train, b_train, weights);
            double rmse_test = computeRMSE(A_test, b_test, weights);
            logFile << epoch << "," << rmse_train << "," << rmse_test << "\n";

            if (epoch % 1000 == 0 || epoch == epochs) {
                std::cout << "Epoch " << std::setw(4) << epoch
                          << " | Train RMSE: " << std::fixed << std::setprecision(4) << std::setw(8) << rmse_train
                          << " | Test RMSE: " << std::fixed << std::setprecision(4) << std::setw(8) << rmse_test
                          << "\n";
            }
        }
    }

    logFile.close();
    std::cout << "Training complete.\n";
    return weights;
}

int main() {
    std::ifstream file("machine.data");
    std::string line;

    std::vector<Vector> allFeatures;
    std::vector<double> allLabels;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        for (int i = 0; i < 2; ++i) getline(ss, token, ',');

        Vector x(6);
        for (int i = 0; i < 6; ++i) {
            getline(ss, token, ',');
            x[i] = std::stod(token);
        }

        getline(ss, token, ',');
        allFeatures.push_back(x);
        allLabels.push_back(std::stod(token));
    }

    normalizeFeatures(allFeatures);

    int total = allFeatures.size();
    std::vector<int> indices(total);
    for (int i = 0; i < total; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(42));

    int trainSize = total * 0.8;
    Matrix A_train(trainSize, 6);
    Vector b_train(trainSize);
    Matrix A_test(total - trainSize, 6);
    Vector b_test(total - trainSize);

    for (int i = 0; i < total; ++i) {
        Vector& x = allFeatures[indices[i]];
        double y = allLabels[indices[i]];

        if (i < trainSize) {
            for (int j = 0; j < 6; ++j)
                A_train(i + 1, j + 1) = x[j];
            b_train[i] = y;
        } else {
            for (int j = 0; j < 6; ++j)
                A_test(i - trainSize + 1, j + 1) = x[j];
            b_test[i - trainSize] = y;
        }
    }

    std::cout << "Total samples read: " << total << "\n";
    Vector weights = gradientDescent(A_train, b_train, A_test, b_test);

    std::cout << "Inference on test set:\n";
    return 0;
}