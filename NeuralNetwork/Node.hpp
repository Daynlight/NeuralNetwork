#pragma once
#include <cstddef>       // for std::size_t
#include <vector>        // for std::vector
#include <string>        // for std::string
#include <sstream>       // for stringstream & ostringstream
#include <stdexcept>     // for std::runtime_error
#include <algorithm>     // for std::copy

namespace NeuralNetwork {

class Node {
private:
  float value = 0.0f;
  std::size_t W = 0;                  // number of input weights (excluding bias)
  std::vector<float> weights;         // size = W + 1, last element is bias weight

public:
  // [Constructors]
  explicit Node(std::size_t inputSize) noexcept 
    : W(inputSize), weights(W + 1, 0.0f) {}

  Node(const std::vector<float>& weights, float value = 0.0f) 
    : W(weights.size() > 0 ? weights.size() - 1 : 0), value(value), weights(weights) {
    if (weights.size() != W + 1)
      throw std::runtime_error("Weights size does not match input size + 1");
  }

  explicit Node(const std::vector<float>& data, bool fromData) {
    if (data.size() < 2) 
      throw std::runtime_error("Data size too small");
    W = data.size() - 2;
    value = data[0];
    weights.assign(data.begin() + 1, data.end());
  }

  explicit Node(const std::string& str) {
    deserialize(str);
  }

  // [Setters/Getters]
  float getValue() const noexcept { return value; }
  const std::vector<float>& getWeights() const noexcept { return weights; }

  void setValue(float v) noexcept { value = v; }
  void setWeights(const std::vector<float>& w) {
    if (w.size() != W + 1)
      throw std::runtime_error("Weights size mismatch");
    weights = w;
  }

  // [Calculate]
  void calculate(const std::vector<float>& previousValues) {
    if (previousValues.size() != W)
      throw std::runtime_error("Input size mismatch for calculate");
    value = 0.0f;
    for (std::size_t i = 0; i < W; ++i)
      value += previousValues[i] * weights[i];
    value += weights[W]; // bias
  }

  // [Get Batch/Load Batch]
  std::vector<float> getData() const noexcept {
    std::vector<float> data(W + 2);
    data[0] = value;
    std::copy(weights.begin(), weights.end(), data.begin() + 1);
    return data;
  }

  void setData(const std::vector<float>& data) {
    if (data.size() != W + 2)
      throw std::runtime_error("Data size mismatch in setData");
    value = data[0];
    std::copy(data.begin() + 1, data.end(), weights.begin());
  }

  // [Serialization/Deserialization]
  std::string serialize() const {
    std::ostringstream str;
    str << value << ";";
    for (float w : weights) str << w << ";";
    return str.str();
  }

  void deserialize(const std::string& str) {
    std::stringstream ss(str);
    std::string token;
    std::vector<float> vals;

    while (std::getline(ss, token, ';')) {
      if (token.empty()) continue;

      try {
        vals.push_back(std::stof(token));
      } catch (const std::exception&) {
        throw std::runtime_error("Invalid float format: " + token);
      }
    }

    if (vals.size() < 2)
      throw std::runtime_error("Not enough data to deserialize");

    value = vals[0];
    W = vals.size() - 2;
    weights.assign(vals.begin() + 1, vals.end());
  }
};

} // namespace NeuralNetwork
