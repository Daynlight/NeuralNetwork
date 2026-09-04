// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "NeuralNetwork.h"



// =========================== //
// ======= Constructors ====== //
// =========================== //
// core
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::NeuralNetwork() noexcept {};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::~NeuralNetwork() noexcept {};



// copy
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::NeuralNetwork(const NeuralNetwork& second) noexcept
  : layers(second.layers) {};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>& NN::NeuralNetwork<First, Second, Rest...>::operator=(const NeuralNetwork& second) noexcept {
  if(this == &second) return *this;

  layers = second.layers;

  return *this;
};



// move
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::NeuralNetwork(NeuralNetwork&& second) noexcept
  : layers(std::move(second.layers)) {};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>& NN::NeuralNetwork<First, Second, Rest...>::operator=(NeuralNetwork&& second) noexcept {
  if(this == &second) return *this;

  layers = std::move(second.layers);

  return *this;
};



// =========================== //
// ======= Constructors ====== //
// =========================== //
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setLearningRate(double learning_rate) noexcept {
  std::apply([&](auto&... layer) {
    (layer.setLearningRate(learning_rate), ...);
  }, layers);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template<std::size_t I, typename T>
inline void NN::NeuralNetwork<First, Second, Rest...>::setActivation() noexcept {
  if constexpr(I < 0) return;
  if constexpr(I >= std::tuple_size_v<LayerTuple>) return;
  std::get<I>(layers).template setActivation<T>();
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template<std::size_t I, typename T>
inline void NN::NeuralNetwork<First, Second, Rest...>::setLoss() noexcept {
  if constexpr(I < 0) return;
  if constexpr(I >= std::tuple_size_v<LayerTuple>) return;
  std::get<I>(layers).template setLoss<T>();
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setInput(std::initializer_list<double> nodes) noexcept {
  std::get<0>(layers).setNodes(nodes);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline double* NN::NeuralNetwork<First, Second, Rest...>::getResult() noexcept {
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return nullptr;
  constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;
  return std::get<last>(layers).getNodes();
};



// =========================== //
// ===== Forward/Backprop ==== //
// =========================== //
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::forward(){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  forwardImpl(std::make_index_sequence<std::tuple_size_v<LayerTuple> - 1>{});
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template <std::size_t... I>
inline void NN::NeuralNetwork<First, Second, Rest...>::forwardImpl(std::index_sequence<I...>){
  ((std::get<I>(layers).forward(std::get<I + 1>(layers))), ...);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backprop(std::initializer_list<double> loss){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  backpropInitial(loss);
  backpropImpl(reverse_sequence(std::make_index_sequence<std::tuple_size_v<LayerTuple> - 1>{}));
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template <std::size_t... I>
inline void NN::NeuralNetwork<First, Second, Rest...>::backpropImpl(std::index_sequence<I...>){
  ((std::get<I>(layers).backprop(std::get<I + 1>(layers))), ...);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backpropInitial(std::initializer_list<double> loss){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;
  std::get<last>(layers).backprop_initial(loss);
};
