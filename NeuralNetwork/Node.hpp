#pragma once
#include <cstddef>       // for std::size_t
#include <array>         // for std::array
#include <string>        // for std::string
#include <sstream>       // for stringstream & ostringstream
#include <stdexcept>     // for std::runtime_error
#include <algorithm>     // for std::copy

namespace NeuralNetwork {
template <std::size_t W>
class Node {
private:
  float value = 0.0f;
  std::array<float, W + 1> weights{};

public:
  // [Constructors]
  constexpr Node() noexcept = default;
  constexpr Node(const std::array<float, W + 1> &weights, const float value = 0) noexcept
    : value(value), weights(weights) {};
  explicit Node(const std::array<float, W + 2> &data) noexcept { setData(data); };
  explicit Node(const std::string &str) { deserialize(str); };

  // [Setters/Getters]
  [[nodiscard]] constexpr float getValue() const noexcept 
    { return value; };
  [[nodiscard]] constexpr const std::array<float, W + 1> &getWeights() const noexcept 
    { return weights; };
  constexpr void setValue(const float value) noexcept
    { this->value = value; };
  constexpr void setWeights(const std::array<float, W + 1> &_weights) noexcept
    { std::copy(_weights.begin(), _weights.end(), weights.begin()); };

  // [Calculate]
  constexpr void calculate(const std::array<float, W> &previousValues) noexcept {
    value = 0.0f;
    for (std::size_t i = 0; i < W; i++) 
      value += previousValues[i] * weights[i];
    value += weights[W];
  };

  // [Get Batch/Load Batch]
  std::array<float, W + 2> getData() const noexcept {
    std::array<float, W + 2> data{};
    data[0] = value;
    for (std::size_t i = 0; i < W + 1; i++) 
      data[i + 1] = weights[i];
    return data;
  };

  void setData(const std::array<float, W + 2> &data) noexcept {
    setValue(data[0]);
    std::copy(data.begin() + 1, data.end(), weights.begin());
  };

  // [Serialization/Deserialization]
  [[nodiscard]] std::string serialize() const {
    std::ostringstream str;
    str << value << ";";
    for (std::size_t i = 0; i < W + 1; i++) str << weights[i] << ";";
    return str.str();
  };

  void deserialize(const std::string &str) {
    std::stringstream ss(str);
    std::string token;
    std::size_t id = 0;

    while (std::getline(ss, token, ';')) {
      if (token.empty()) continue;

      float val;
      try {
        val = std::stof(token);  // Convert token to float
      } catch (const std::exception &) {
        throw std::runtime_error("Invalid float format: " + token);
      }

      if (id == 0) setValue(val);
      else if (id <= W + 1) weights[id - 1] = val;
      else throw std::runtime_error("Too many values in input string");

      id++;
    };

    if (id != W + 2)
      throw std::runtime_error("Incorrect number of tokens in data string");
  };
};
}; // namespace NeuralNetwork
