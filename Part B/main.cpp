#include "LinearRegression.h"
#include <iostream>

using namespace std;

int main() {
    LinearRegression regression;
    string filename = "machine.data";

    // Load and normalize data
    readData(regression, filename);

    // Split dataset
    cout << "Splitting dataset..." << endl;
    regression.splitData();  // default is 80% train, 20% test

    // Train model
    cout << "Starting training..." << endl;
    regression.train(5000, 1e-3, 5);  // epochs, learning rate, batch size

    // Inference
    cout << "Training complete.\nInference on test set:\n";
    const vector<pair<double,double>>& predictions = regression.infer();
    for (size_t i = 0; i < predictions.size(); ++i) {
        cout << "Pred and GT for sample " << i + 1 << ": "
             << predictions[i].first << " "
             << predictions[i].second << endl;
    }

    // Export RMSE log
    exportRMSEToCSV("rmse_log.csv",
                    regression.getTrainRMSEs(),
                    regression.getTestRMSEs());

    // Show weights
    cout << "Final weights: ";
    const vector<double>& weights = regression.getWeights();
    for (const auto& w : weights) {
        cout << w << " ";
    }
    cout << "\n";

    return 0;
}