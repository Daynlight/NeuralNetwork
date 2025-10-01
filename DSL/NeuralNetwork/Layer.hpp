#include "Layer.h"

template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::Layer() noexcept {
  for(double &el : weights) 
    el = ((rand() % 2000) / 1000.0 - 1) * 0.1;

  loss = getLossFromType(LossType::MSETYPE);
  activation = getActivationFromType(ActivationType::LINEARTYPE);
}

template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::~Layer() noexcept {
  delete loss;
  delete activation;
}

template <unsigned int S, unsigned int D>
inline double *NN::Layer<S, D>::getNodes() noexcept {
  return nodes;
}

template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setNodes(std::initializer_list<double> nodes) noexcept {
  unsigned int i = 0;
  for (auto it = nodes.begin(); it != nodes.end() && i < S; ++it, ++i)
    this->nodes[i] = *it;
}

template <unsigned int S, unsigned int D>
inline double *NN::Layer<S, D>::getWeights() noexcept {
  return weights;
}

template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(std::initializer_list<double> weights) noexcept {
  unsigned int i = 0;
  for (auto it = weights.begin(); it != weights.end() && i < (S + 1) * D; ++it, ++i)
    this->weights[i] = *it;
}

template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(const double *weights) {
  for (unsigned int i = 0; i < (S + 1) * D; ++i) {
    this->weights[i] = weights[i];
  }
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
inline void NN::Layer<S, D>::forward(NN::Layer<D, N> &layer) {
  nodes[S] = 1.0; // bias
  for(unsigned int i = 0; i < D; i++){
      double sum = 0;
      for(unsigned int j = 0; j < S + 1; j++)
          sum += nodes[j] * weights[i * (S + 1) + j];
      layer[i] = activation ? activation->fun(sum) : sum;
  }
}

template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop_initial(Layer<N, S> &layer, 
std::initializer_list<double> target) noexcept {
  unsigned int i = 0;
  for (auto it = target.begin(); it != target.end() && i < S; ++it, ++i){
    if(activation)
      sigma[i] = loss->fun_prime(activation->fun(nodes[i]), *it) 
                * activation->fun_prime(nodes[i]);
    else
      sigma[i] = loss->fun_prime(nodes[i], *it);
  }
  
  double *weights_back = layer.getWeights();
  for(unsigned int j = 0; j < S; j++)
    for(unsigned int i = 0; i < N + 1; i++)
      weights_back[j * (S + 1) + i] -= learning_rate * layer[i] * sigma[j];
  
  layer.setWeights(weights_back);
};

template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop(Layer<D, N> &next_layer) noexcept {
    const double* sigma_next = next_layer.getSigma();

    for(unsigned int i = 0; i < S; ++i){
      double sum = 0;
      for(unsigned int j = 0; j < D; ++j){
          sum += weights[j * (S + 1) + i] * sigma_next[j];
      }
      sigma[i] = activation ? sum * activation->fun_prime(nodes[i]) : sum;
    }

    double* weights_back = getWeights();
    for(unsigned int j = 0; j < D; ++j){
        for(unsigned int i = 0; i < S; ++i){
            weights_back[j * (S + 1) + i] -= learning_rate * nodes[i] * sigma_next[j];
        }
        weights_back[j * (S + 1) + S] -= learning_rate * 1.0 * sigma_next[j];
    }
    setWeights(weights_back);
};


template <unsigned int S, unsigned int D>
inline const double *NN::Layer<S, D>::getSigma() const noexcept {
  return sigma;
}

template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::print() const
{
  std::string s = "size: \n";

  s += std::to_string(S) + ", ";
  s += std::to_string(D) + "; ";

  s += "\nloss, activation \n";
  s += std::to_string(loss->getType()) + "; ";
  if(activation)
    s += std::to_string(activation->getType()) + "; ";
  else
    s += "-1; ";


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
inline double &NN::Layer<S, D>::operator[](unsigned int i) {
  if(i > S) 
    throw std::range_error("index out of range");
  return nodes[i];
}
