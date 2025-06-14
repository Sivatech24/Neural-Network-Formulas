#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

std::vector<double> loadInput(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<double> data;
    double val;
    while (file >> val) {
        data.push_back(val);
    }
    return data;
}

double relu(double x) {
    return x > 0 ? x : 0;
}

std::vector<double> forwardPass(const std::vector<double>& input, const std::vector<double>& weights, double bias) {
    std::vector<double> output;
    for (size_t i = 0; i < input.size(); ++i) {
        double z = input[i] * weights[i] + bias;
        output.push_back(relu(z));
    }
    return output;
}

void visualizeOutput(const std::vector<double>& output) {
    for (double val : output) {
        int color = static_cast<int>(val * 255);
        if (color > 255) color = 255;
        if (color < 0) color = 0;
        std::cout << "\033[48;2;" << color << ";" << 255 - color << ";100m  \033[0m";
    }
    std::cout << std::endl;
}

int main() {
    auto input = loadInput("data.txt");

    // Dummy weights and bias
    std::vector<double> weights(input.size(), 0.5);
    double bias = 0.1;

    auto output = forwardPass(input, weights, bias);
    visualizeOutput(output);

    return 0;
}
