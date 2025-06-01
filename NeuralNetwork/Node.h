#pragma once
#include <stddef.h>
#include <array>
#include <string>
#include <sstream>


namespace NeuralNetwork
{
template <size_t W>
class Node{
private:
  float value = 0.0f;
  std::array<float, W> weights{};
  float bias = 0.0f;

public:
  // [Constructors]
  Node() = default;
  Node(const std::array<float, W> &weights, const float bias, const float value) { setValue(value); setBias(bias); setWeights(weights); };
  Node(const std::array<float, W + 2> &data) { setData(data); };
  explicit Node(const std::string &str) { setDataString(str); };
  
  // [Setters/Getters]
  float getValue() const { return value; };
  float getBias() const { return bias; };
  const std::array<float, W> &getWeights() const { return weights; };
  void setValue(const float value) { this->value = value; };
  void setBias(const float bias) { this->bias = bias; };
  void setWeights(const std::array<float, W> &weights) { this->weights = weights; };

  // [Calculate]
  void calculate(const std::array<float, W> &previousValues) { 
    value = 0.0f;
    for(size_t i = 0; i < W; i++) value += previousValues[i] * weights[i]; 
    value += bias;
  };

  // [Get Batch/Load Batch]
  std::array<float, W + 2> getData() const {
    std::array<float, W + 2> data{};
    data[0] = value;
    data[1] = bias;
    for (size_t i = 0; i < W; i++) {
        data[i + 2] = weights[i];
    }
    return data;
  };
  void setData(const std::array<float, W + 2> &data) {
    setValue(data[0]);
    setBias(data[1]);
    for (size_t i = 0; i < W; i++) {
        weights[i] = data[i + 2];
    }
  };

  // [Save/Load]
  std::string getDataString() const {
    std::string str;
    str += std::to_string(value) + ";";
    str += std::to_string(bias) + ";";
    for (size_t i = 0; i < W; i++) str += std::to_string(weights[i]) + ";";
    return str;
  };
  void setDataString(const std::string& str) {
      std::stringstream ss(str);
      std::string token;
      size_t id = 0;
      while (std::getline(ss, token, ';')) {
          if (token.empty()) continue;
          float fvalue = std::stof(token);
          if (id == 0) setValue(fvalue);
          else if (id == 1) setBias(fvalue);
          else if (id - 2 < W) weights[id - 2] = fvalue;
          else throw std::runtime_error("Too many tokens");
          id++;
      };
      if (id != W + 2) throw std::runtime_error("Too few tokens in data string");
  };
};
}; // namespace NeuralNetwork
