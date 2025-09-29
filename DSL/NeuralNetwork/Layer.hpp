#include "Layer.h"

template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::Layer() noexcept {
  for(double &el : nodes)
    el = double(1);
  for(double &el : weights)
    el = double(1);
  loss = new MSE();
}

template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::~Layer() noexcept {
  delete loss;
  delete activation;
}

template <unsigned int S, unsigned int D>
inline const double *&NN::Layer<S, D>::getNodes() const noexcept {
  return nodes;
}

template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setNodes(std::initializer_list<double> nodes) noexcept {
  unsigned int i = 0;
  for (auto it = nodes.begin(); it != nodes.end() && i < S; ++it, ++i)
    this->nodes[i] = *it;
}

template <unsigned int S, unsigned int D>
inline const double *&NN::Layer<S, D>::getWeights() const noexcept {
  return weights;
}

template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(std::initializer_list<double> weights) noexcept {
  unsigned int i = 0;
  for (auto it = weights.begin(); it != weights.end() && i < (S + 1) * D; ++it, ++i)
    this->weights[i] = *it;
}

template <unsigned int S, unsigned int D>
inline const NN::iActivation *NN::Layer<S, D>::getActivation() const noexcept {
  return activation;
}

template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setActivation(ActivationType type) noexcept {
  delete activation;
  activation = getActivationFromType(type);
}

template <unsigned int S, unsigned int D>
inline const NN::iLoss *NN::Layer<S, D>::getLoss() const noexcept {
  return loss;
}

template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setLoss(LossType type) noexcept {
  delete loss;
  loss = getLossFromType(type);
}

template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::forward(NN::Layer<D, N> &layer) noexcept {
  for(unsigned int i = 0; i < D; i++){
    double sum = 0;
    for(unsigned int j = 0; j < S + 1; j++)
      sum += this->nodes[j] * this->weights[i * (S + 1) + j];
    layer[i] = sum;
  };
};

template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::print() const {
  std::string s = "";

  s += std::to_string(S) + ", ";
  s += std::to_string(D) + "; ";

  s += std::to_string(loss->getType()) + "; ";
  if(activation)
    s += std::to_string(activation->getType()) + "; ";
  else
    s += "-1; ";


  bool first = true;
  s += "[";
  for(double el : nodes){
    if(!first) 
      s += ", ";
      s += std::to_string(el);
      first = false;
  }
  s += "]; ";
  
  first = true;
  unsigned int i = 0;
  s += "[[";
  for(double el : weights){
    if(!first && i % (S + 1) != 0)
      s += ", ";

    if(i % (S + 1) == 0 && i <= (S + 1) * D && i != 0) 
      s += "], [";

    s += std::to_string(el);
    first = false;
    i++;
  };
  s += "]]; ";
  return s;
}

template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::serialize() const {
  std::string s = "";

  s += std::to_string(S) + ", ";
  s += std::to_string(D) + "; ";

  s += std::to_string(loss->getType()) + "; ";
  if(activation)
    s += std::to_string(activation->getType()) + "; ";
  else
    s += "-1; ";

  bool first = true;
  for(double el : nodes){
    if(!first) 
      s += ", ";
      s += std::to_string(el);
      first = false;
  }
  s += "; ";
  
  first = true;
  for(double el : weights){
    if(!first) 
      s += ", ";
      s += std::to_string(el);
      first = false;
  };
  s += "; ";
  return s;
}

template <unsigned int S, unsigned int D>
inline double &NN::Layer<S, D>::operator[](unsigned int i){
  return nodes[i];
}
