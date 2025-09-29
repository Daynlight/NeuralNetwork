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
  Layer() noexcept;
  Layer(const double &nodes, 
        const double &weights, 
        iLoss * loss = new MSE(), 
        iActivation iActivation = nullptr);
  ~Layer() noexcept;
  
  const double* &getNodes() const noexcept;
  void setNodes(double nodes[S + 1]) noexcept;
  
  const double* &getWeights() const noexcept;
  void setWeights(double weights[(S + 1) * D]) noexcept;

  const iActivation *getActivation() const noexcept;
  void setActivation(const iActivation *activation) noexcept;

  const iLoss *getLoss() const noexcept;
  void setLoss(const iLoss *loss) noexcept;

  template<unsigned int N>
  void forward(Layer<D, N> layer);

  template<unsigned int N>
  void backprop(Layer<N, S> layer);

  constexpr std::string serialize() const;
};
};

#include "Layer.hpp"