#pragma once
#include "Activation.h"
#include "Loss.h"
#include <string>

namespace NN{
template<unsigned int S, unsigned int D>
class Layer{
private:
  double nodes[S + 1];
  double weights[(S + 1) * D];
  iActivation *activation = nullptr;
  iLoss *loss = nullptr;

public:
  Layer() noexcept;   // O(n)
  ~Layer() noexcept;  // O(1)
  
  const double* &getNodes() const noexcept;   // O(1)
  void setNodes(std::initializer_list<double> nodes) noexcept;    // O(n)
  
  const double* &getWeights() const noexcept;   // O(1)
  void setWeights(std::initializer_list<double> weights) noexcept;    // O(n)

  const iActivation *getActivation() const noexcept;    // O(1)
  void setActivation(ActivationType type) noexcept;   // O(1)

  const iLoss *getLoss() const noexcept;    // O(1)
  void setLoss(LossType type) noexcept;   // O(1)

  template<unsigned int N>
  void forward(Layer<D, N> layer) noexcept;

  template<unsigned int N>
  void backprop(Layer<N, S> layer) noexcept;

  std::string print() const;    // O(n)
  std::string serialize() const;    // O(n)
  void deserialize(const std::string &data);
};
};

#include "Layer.hpp"