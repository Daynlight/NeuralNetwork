#pragma once
#include "Activation/Activation.h"
#include "Loss/Loss.h"
#include <string>
#include <stdexcept>

namespace NN {
//----------------------------------------------------------------//
//------------------------ Representation ------------------------//
//----------------------------------------------------------------//
// Nodes are S + 1 double Array where last one is always 1 for bias
// todo: will be changed to Math:Vec in feature
//----------------------------------------------------------------//
// Weights is matrix S + 1 Columns and D Rows, D is 1 less then 
// second layer nodes because last node is always 1 and we don't
// update it
// todo: will be change to Math::Matrix in feature
//----------------------------------------------------------------//
// Loss is interfaced to be function and prime function look Loss.h
// always setted default MSE f(x) = ((x - t)^2)/2, f'(x) = x - t
// todo: change from pointer to function type
//----------------------------------------------------------------//
// Activation is interfaced to be function and prime function
// look Activation.h always setted default Linear f(x) = x, f'(x) = 1
// todo: change from pointer to function type
//----------------------------------------------------------------//
// Learning rate indicate learning speed
// todo: will changed to modes default static and dynamic
//----------------------------------------------------------------//

template<unsigned int S, unsigned int D>
class Layer{
private:
  double nodes[S + 1];
  double sigma[S];
  double weights[(S + 1) * D];
  iActivation *activation = nullptr;
  iLoss *loss = nullptr;
  double learning_rate = 0.005;

public:
  Layer() noexcept;   // O((S+1) * D)
  ~Layer() noexcept;  // O(1)
  
  double* getNodes() noexcept;   // O(1)
  void setNodes(std::initializer_list<double> nodes) noexcept;    // O(n)
  double getActivatedNode(unsigned int i);    // O(1)
  
  double* getWeights() noexcept;   // O(1)
  void setWeights(std::initializer_list<double> weights) noexcept;    // O(n)
  void setWeights(const double *weights);

  double getLearningRate() const noexcept;    // O(1)
  void setLearningRate(double learning_rate);    // O(1)

  const iActivation *getActivation() const noexcept;    // O(1)
  void setActivation(ActivationType type) noexcept;   // O(1)

  const iLoss *getLoss() const noexcept;    // O(1)
  void setLoss(LossType type) noexcept;   // O(1)

  template<unsigned int N>
  void forward(Layer<D, N> &layer);   // O(n^2)

  template<unsigned int N>
  void backprop_initial(Layer<N, S> &layer, std::initializer_list<double> target) noexcept;
  template<unsigned int N>
  void backprop(Layer<D, N> &layer) noexcept;
  const double *getSigma() const noexcept;    // O(1)

  std::string print() const;    // O(n)
  std::string serialize() const;    // O(n)
  // todo: write deserialize method
  void deserialize(const std::string &data);

  double &operator[](unsigned int i);   // O(1)
};
};

#include "Layer.hpp"