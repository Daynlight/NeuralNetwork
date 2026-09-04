// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Layer.h"



// =========================== //
// ======= Constructors ====== //
// =========================== //
// core
template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::Layer() noexcept {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  
  #pragma omp parallel for
  for(double &el : weights) el = dist(gen);

  loss = std::make_unique<NN::MSE>();
  activation = std::make_unique<NN::Linear>();
};



template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::~Layer() noexcept { };



// ============================== //
// ======= Setters/Getters ====== //
// ============================== //
template <unsigned int S, unsigned int D>
inline double *NN::Layer<S, D>::getNodes() noexcept {
  return nodes;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setNodes(std::initializer_list<double> nodes) noexcept {
  unsigned int i = 0;
  #pragma omp parallel for
  for (auto it = nodes.begin(); it != nodes.end() && i < S; ++it, ++i) this->nodes[i] = *it;
};



template <unsigned int S, unsigned int D>
inline double NN::Layer<S, D>::getActivatedNode(unsigned int i) const noexcept {
  return activation->fun(nodes[i]);
};



template <unsigned int S, unsigned int D>
inline double *NN::Layer<S, D>::getWeights() noexcept {
  return weights;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(std::initializer_list<double> weights) noexcept {
  unsigned int i = 0;
  #pragma omp parallel for
  for (auto it = weights.begin(); it != weights.end() && i < (S + 1) * D; ++it, ++i) this->weights[i] = *it;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(const double* weights) {
  #pragma omp parallel for
  for (unsigned int i = 0; i < (S + 1) * D; ++i) this->weights[i] = weights[i];
};



template <unsigned int S, unsigned int D>
inline double NN::Layer<S, D>::getLearningRate() const noexcept {
  return learning_rate;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setLearningRate(double learning_rate) noexcept {
  this->learning_rate = learning_rate;
};



template <unsigned int S, unsigned int D>
inline const std::unique_ptr<NN::iActivation>& NN::Layer<S, D>::getActivation() const noexcept {
  return activation;
};



template <unsigned int S, unsigned int D>
template<typename T>
inline void NN::Layer<S, D>::setActivation() noexcept {
  activation = std::make_unique<T>();
};



template <unsigned int S, unsigned int D>
inline const std::unique_ptr<NN::iLoss>& NN::Layer<S, D>::getLoss() const noexcept {
  return loss;
};



template <unsigned int S, unsigned int D>
template<typename T>
inline void NN::Layer<S, D>::setLoss() noexcept {
  loss = std::make_unique<T>();
};



template <unsigned int S, unsigned int D>
inline double &NN::Layer<S, D>::operator[](unsigned int i) {
  if(i > S) throw std::range_error("index out of range");
  return nodes[i];
};



template <unsigned int S, unsigned int D>
inline const double *NN::Layer<S, D>::getSigma() const noexcept {
  return sigma;
};



// =============================== //
// ======= Forward/Backprop ====== //
// =============================== //
template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::forward(NN::Layer<D, N> &layer) {
  nodes[S] = 1.0; // bias
  
  #pragma omp parallel for
  for(unsigned int i = 0; i < D; i++){
    double sum = 0;
    #pragma omp parallel for
    for(unsigned int j = 0; j < S + 1; j++){
      if(j != S) sum += getActivatedNode(j) * weights[i * (S + 1) + j];
      else sum += weights[i * (S + 1) + j];
    };
    layer[i] = sum;
  };

  layer[D] = 1;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial(std::initializer_list<double> target) noexcept {
  unsigned int i = 0;
  #pragma omp parallel for
  for(auto it = target.begin(); it != target.end() && i < S; ++it, ++i)
    sigma[i] = loss->fun_prime(getActivatedNode(i), *it) * activation->fun_prime(nodes[i]);
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop(Layer<D, N> &next_layer) noexcept {
  const double* sigma_next = next_layer.getSigma();

  #pragma omp parallel for
  for(unsigned int i = 0; i < S; ++i){
    double sum = 0;
    #pragma omp parallel for
    for(unsigned int j = 0; j < D; ++j) sum += weights[j * (S + 1) + i] * sigma_next[j];
    sigma[i] = sum * activation->fun_prime(nodes[i]);
  };

  double* weights_back = getWeights();
  #pragma omp parallel for
  for(unsigned int j = 0; j < D; ++j){
    #pragma omp parallel for
    for(unsigned int i = 0; i < S; ++i)
      weights_back[j * (S + 1) + i] -= learning_rate * getActivatedNode(i) * sigma_next[j];
    weights_back[j * (S + 1) + S] -= learning_rate * 1.0 * sigma_next[j];
  };
};



// =========================== //
// ======= Presentation ====== //
// =========================== //
template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::print() const {
  std::string s = "size: \n";

  s += std::to_string(S) + ", ";
  s += std::to_string(D) + "; ";

  // s += "\nloss, activation \n";
  // s += std::to_string(loss->getType()) + "; ";
  // if(activation)
    // s += std::to_string(activation->getType()) + "; ";
  // else
    // s += "-1; ";


  bool first = true;
  s += "\nnodes: \n[";
  for(double el : nodes){
    if(!first) 
      s += ", ";
      s += std::to_string(el);
      first = false;
  }
  s += "]; ";
  
  first = true;
  unsigned int i = 0;
  s += "\nweights: \n[[";
  for(double el : weights){
    if(!first && i % (S + 1) != 0)
      s += ", ";

    if(i % (S + 1) == 0 && i <= (S + 1) * D && i != 0) 
      s += "], \n[";

    s += std::to_string(el);
    first = false;
    i++;
  };
  s += "]];\n ";
  return s;
};
