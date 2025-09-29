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
inline const NN::iActivation *NN::Layer<S, D>::getActivation() const noexcept {
  return activation;
}

template <unsigned int S, unsigned int D>
inline const NN::iLoss *NN::Layer<S, D>::getLoss() const noexcept {
  return loss;
}

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