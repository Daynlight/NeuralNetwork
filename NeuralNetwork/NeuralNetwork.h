#pragma once

#include "Layer/Layer.h"
#include <tuple>

namespace NN {

template<unsigned int... L>
class NeuralNetwork;
template<unsigned int First, unsigned int Second, unsigned int... Rest>
class NeuralNetwork<First, Second, Rest...> {
private:
  std::tuple<
    Layer<First, Second>,
    Layer<Second, Rest>...
  > layers;

public:
  NeuralNetwork() = default;
  void setLearningRate(double learning_rate);
  template<std::size_t I>
  void setActivation(ActivationType type);
  void setNodes(std::initializer_list<double> nodes);
  void forward();
  template <std::size_t... I>
  void forwardImpl(std::index_sequence<I...>);
  void backprop(std::initializer_list<double> loss);
  template <std::size_t... I>
  void backpropImpl(std::index_sequence<I...>);
  void backpropInitial(std::initializer_list<double> loss);

  double* getResult();
};

};

#include "NeuralNetwork.hpp"