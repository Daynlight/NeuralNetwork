#pragma once

#include "Layer/Layer.h"
#include <tuple>
#include <utility>

namespace NN {
template <unsigned int... Ns>
struct LayerBuilder;

template <unsigned int A, unsigned int B, unsigned int... Rest>
struct LayerBuilder<A, B, Rest...> {
  using type = decltype(std::tuple_cat(
    std::tuple<Layer<A, B>>{},
    typename LayerBuilder<B, Rest...>::type{}
  ));
};
template <unsigned int A, unsigned int B>
struct LayerBuilder<A, B> {
  using type = std::tuple<Layer<A, B>>;
};

template <std::size_t... I>
constexpr auto reverse_sequence(std::index_sequence<I...>) {
  return std::index_sequence<sizeof...(I) - 1 - I...>{};
}

template<unsigned int... L>
class NeuralNetwork;

template<unsigned int First, unsigned int Second, unsigned int... Rest>
class NeuralNetwork<First, Second, Rest...> {
private:
  using LayerTuple = typename LayerBuilder<First, Second, Rest...>::type;

  LayerTuple layers;

public:
  NeuralNetwork() = default;

  void setLearningRate(double learning_rate) {
    std::apply([&](auto&... layer) {
      (layer.setLearningRate(learning_rate), ...);
    }, layers);
  }

  template<std::size_t I>
  void setActivation(ActivationType type) {
    std::get<I>(layers).setActivation(type);
  }

  void setNodes(std::initializer_list<double> nodes) {
    std::get<0>(layers).setNodes(nodes);
  }

  void forward() {
    forwardImpl(std::make_index_sequence<std::tuple_size_v<LayerTuple> - 1>{});
  }

  template <std::size_t... I>
  void forwardImpl(std::index_sequence<I...>) {
    ((std::get<I>(layers).forward(std::get<I + 1>(layers))), ...);
  }

  void backprop(std::initializer_list<double> loss) {
    backpropInitial(loss);

    backpropImpl(reverse_sequence(std::make_index_sequence<std::tuple_size_v<LayerTuple> - 1>{}));
  }

  template <std::size_t... I>
  void backpropImpl(std::index_sequence<I...>) {
    ((std::get<I>(layers).backprop(std::get<I + 1>(layers))), ...);
  }

  void backpropInitial(std::initializer_list<double> loss) {
    constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;

    std::get<last>(layers)
      .backprop_initial(std::get<last - 1>(layers), loss);
  }

  double* getResult() {
    constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;
    return std::get<last>(layers).getNodes();
  }
};

} // namespace NN