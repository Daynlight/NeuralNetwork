#include "NeuralNetwork.h"
#include <iostream>

template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setLearningRate(double learning_rate){
  std::apply([&](auto&... layer) {
    (layer.setLearningRate(learning_rate), ...);
  }, layers);
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
template<std::size_t I>
inline void NN::NeuralNetwork<First, Second, Rest...>::setActivation(ActivationType type)
{
  std::get<I>(layers).setActivation(type);
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setNodes(std::initializer_list<double> nodes)
{
  std::get<0>(layers).setNodes(nodes);
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::forward()
{
  forwardImpl(std::make_index_sequence<sizeof...(Rest)>{});
}

template <std::size_t... I>
constexpr auto reverse(std::index_sequence<I...>)
{
  return std::index_sequence<sizeof...(I) - 1 - I...>{};
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backprop(std::initializer_list<double> loss)
{
  backpropInitial(loss);
  backpropImpl(reverse(std::make_index_sequence<sizeof...(Rest)>{}));
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
template <std::size_t... I>
inline void NN::NeuralNetwork<First, Second, Rest...>::backpropImpl(std::index_sequence<I...>)
{
  ((std::get<I>(layers).backprop(std::get<I + 1>(layers))), ...);
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backpropInitial(std::initializer_list<double> loss)
{
  return std::get<sizeof...(Rest)>(layers).backprop_initial(std::get<sizeof...(Rest) - 1>(layers), loss);
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline double *NN::NeuralNetwork<First, Second, Rest...>::getResult()
{
  return std::get<sizeof...(Rest)>(layers).getNodes();
}

template <unsigned int First, unsigned int Second, unsigned int... Rest>
template <std::size_t... I>
inline void NN::NeuralNetwork<First, Second, Rest...>::forwardImpl(std::index_sequence<I...>)
{
  ((std::get<I>(layers).forward(std::get<I + 1>(layers))), ...);
}