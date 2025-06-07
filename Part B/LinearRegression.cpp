#include "LinearRegression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;

LinearRegression::LinearRegression() {
    weights = vector<double>(6, rand() % 400);
    bias = rand() % 400;
}

void LinearRegression::addData(const vector<double>& sample) {
    if (sample.size() != 7) {
        cerr << "Error: Each input vector must have exactly 7 elements." << endl;
        return;
    }
    data.push_back(sample);
}

int LinearRegression::getLength() const {
    return data.size();
}

const std::vector<double>& LinearRegression::getWeights() const {
    return weights;
}

void LinearRegression::splitData(double trainRatio) {
    vector<vector<double>> shuffled = data;
    random_device rd;
    mt19937 g(rd());
    shuffle(shuffled.begin(), shuffled.end(), g);

    int trainSize = static_cast<int>(shuffled.size() * trainRatio);
    trainSet.assign(shuffled.begin(), shuffled.begin() + trainSize);
    testSet.assign(shuffled.begin() + trainSize, shuffled.end());
}

double LinearRegression::computeRMSE(const vector<vector<double>>& dataset) {
    double totalError = 0.0;
    for (const auto& sample : dataset) {
        double pred = predict(sample);
        double actual = sample[6];
        double err = pred - actual;
        totalError += err * err;
    }
    return sqrt(totalError / dataset.size());
}

double LinearRegression::predict(const vector<double>& sample) {
    double result = bias;
    for (int i = 0; i < 6; ++i)
        result += weights[i] * sample[i];
    return result;
}

void LinearRegression::train(int epochs, double learning_rate, int batch_size) {
    if (trainSet.empty()) {
        cerr << "Error: Train set is empty. Call splitData() first." << endl;
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        random_device rd;
        mt19937 g(rd());
        shuffle(trainSet.begin(), trainSet.end(), g);

        for (size_t i = 0; i < trainSet.size(); i += batch_size) {
            vector<double> gradW(6, 0.0);
            double gradB = 0.0;

            int actual_batch_size = min(batch_size, (int)(trainSet.size() - i));
            for (int j = 0; j < actual_batch_size; ++j) {
                const auto& sample = trainSet[i + j];
                double y_pred = predict(sample);
                double y_true = sample[6];
                double error = y_pred - y_true;

                for (int k = 0; k < 6; ++k)
                    gradW[k] += error * sample[k];
                gradB += error;
            }

            for (int k = 0; k < 6; ++k)
                weights[k] -= learning_rate * (gradW[k] / actual_batch_size);
            bias -= learning_rate * (gradB / actual_batch_size);
        }

        // NEW: Log RMSE for plot
        double train_rmse = computeRMSE(trainSet);
        double test_rmse = computeRMSE(testSet);
        trainRMSEs.push_back(train_rmse);
        testRMSEs.push_back(test_rmse);

        if (epoch % 500 == 0 || epoch == epochs - 1) {
            cout << "Epoch " << epoch
                 << " | Train RMSE: " << train_rmse
                 << " | Test RMSE: " << test_rmse << endl;
        }
    }
}

const vector<pair<double, double>> LinearRegression::infer() {
    vector<pair<double, double>> predictions;
    for (const auto& sample : testSet) {
        double pred = predict(sample);
        predictions.emplace_back(pred, sample[6]);
    }
    return predictions;
}

const vector<double>& LinearRegression::getTrainRMSEs() const {
    return trainRMSEs;
}

const vector<double>& LinearRegression::getTestRMSEs() const {
    return testRMSEs;
}

// Optional CSV export
void exportRMSEToCSV(const string& filename,
                     const vector<double>& trainRMSEs,
                     const vector<double>& testRMSEs) {
    ofstream file(filename);
    file << "Epoch,TrainRMSE,TestRMSE\n";
    for (size_t i = 0; i < trainRMSEs.size(); ++i) {
        file << i << "," << trainRMSEs[i] << "," << testRMSEs[i] << "\n";
    }
    file.close();
}

void readData(LinearRegression& regression, const string& filename) {
    ifstream infile(filename);
    string line;
    int cnt = 0;

    vector<vector<int>> rawSamples;

    while (getline(infile, line)) {
        stringstream ss(line);
        string word;
        vector<int> sample;

        for (int i = 0; i < 10; ++i) {
            if (!getline(ss, word, ',')) break;
            if (i >= 2 && i <= 8) sample.push_back(stoi(word));
        }

        if (sample.size() == 7) {
            rawSamples.push_back(sample);
            cnt++;
        }
    }

    cout << "Total samples read: " << cnt << endl;

    vector<int> minVals(7, numeric_limits<int>::max());
    vector<int> maxVals(7, numeric_limits<int>::min());

    for (const auto& sample : rawSamples) {
        for (int i = 0; i < 7; ++i) {
            minVals[i] = min(minVals[i], sample[i]);
            maxVals[i] = max(maxVals[i], sample[i]);
        }
    }

    for (const auto& sample : rawSamples) {
        vector<double> normalized(7);
        for (int i = 0; i < 6; ++i) {
            if (maxVals[i] == minVals[i])
                normalized[i] = 0.0;
            else
                normalized[i] = (sample[i] - minVals[i]) / double(maxVals[i] - minVals[i]);
        }
        normalized[6] = sample[6];  // Ground truth
        regression.addData(normalized);
    }
}