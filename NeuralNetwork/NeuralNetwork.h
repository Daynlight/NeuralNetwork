#pragma once
#include <stddef.h>

namespace NeuralNetwork
{
template <typename T, size_t W>
class Node{
private:
  T value = 0.0f;
  T weights[W];

public:
  Node(const T weights[W] = nullptr){
    if(weights) 
      for(size_t i = 1; i < W; i++) { this->weights[i] = weights[i]; }
    else 
      for(T &el : weights) { el = 0.0f; };
  }
};
} // namespace NeuralNetwork
