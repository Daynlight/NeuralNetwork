#pragma once
#include <stddef.h>
#include <array>
#include <string>
#include <sstream>

// [TODO] save values in global huge array and contain only id of this array and reference to Array
// [TODO] edge case exceptions
// [TODO] use char_from instead of std::stof

namespace NeuralNetwork
{
template <size_t W>
class Node{
private:
  float value = 0.0f;
  std::array<float, W + 1> weights{};

public:
  // [Constructors]
  Node() = default;
  Node(const std::array<float, W + 1> &weights, const float value = 0) 
    { setValue(value); setWeights(weights); };
  Node(const std::array<float, W + 2> &data) { setData(data); };
  explicit Node(const std::string &str) { deserialize(str); };
  
  // [Setters/Getters]
  float getValue() const { return value; };
  const std::array<float, W + 1> &getWeights() const { return weights; };
  void setValue(const float value) { this->value = value; };
  void setWeights(const std::array<float, W + 1> &weights) { this->weights = weights; };

  // [Calculate]
  void calculate(const std::array<float, W> &previousValues) { 
    value = 0.0f;
    for(size_t i = 0; i < W; i++) value += previousValues[i] * weights[i]; 
    value += weights[W];
  };

  // [Get Batch/Load Batch]
  std::array<float, W + 2> getData() const {
    std::array<float, W + 2> data{};
    data[0] = value;
    for (size_t i = 0; i < W + 1; i++) data[i + 1] = weights[i];
    return data;
  };

  void setData(const std::array<float, W + 2> &data) {
    setValue(data[0]);
    for (size_t i = 0; i < W + 1; i++) weights[i] = data[i + 1];
  };

  // [Serialization/Deserialization]
  std::string serialize() const {
    std::ostringstream str;
    str << value << ";";
    for (size_t i = 0; i < W + 1; i++) str << weights[i] << ";";
    return str.str();
  };

  void deserialize(const std::string& str) {
      std::stringstream ss(str);
      std::string token;
      size_t id = 0;
      while (std::getline(ss, token, ';')) {
          if (token.empty()) continue;

          float fvalue;
          try { fvalue = std::stof(token); }
          catch (const std::invalid_argument& e)
            { throw std::runtime_error("Invalid float: '" + token + "'"); }
          
          if (id == 0) setValue(fvalue);
          else if (id <= W + 1) weights[id - 1] = fvalue;
          id++;
      };
      if (id != W + 2) throw std::runtime_error("Incorrect number of tokens in data string");
  };
};
}; // namespace NeuralNetwork
