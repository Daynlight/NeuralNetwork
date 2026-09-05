// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Activation.h"



// ===================== //
// ======= Linear ====== //
// ===================== //
inline double NN::Linear::fun(std::span<const double> layer, size_t i) const noexcept {
  return layer[i];
};



inline double NN::Linear::fun_prime(std::span<const double> layer, size_t i, size_t j) const noexcept {
  return i == j ? 1.0 : 0.0; 
};



// ====================== //
// ======= Sigmoid ====== //
// ====================== //
inline double NN::Sigmoid::fun(std::span<const double> layer, size_t i) const noexcept {
  if(layer[i] >= 0) return 1.0 / (1.0 + std::exp(-(layer[i])));
  else {
    double z = std::exp(layer[i]);
    return z / (1.0 + z);
  };
};



inline double NN::Sigmoid::fun_prime(std::span<const double> layer, size_t i, size_t j) const noexcept {
  if(i != j) return 0.0;
  double y = fun(layer, i);
  return y * (1 - y); 
};



// ====================== //
// ======= Softmax ====== //
// ====================== //
inline double NN::Softmax::fun(std::span<const double> layer, size_t i) const noexcept {
  double max = *std::max_element(layer.begin(), layer.end());
  double sum = 0.0;

  for(double x : layer) sum += std::exp(x - max);

  return std::exp(layer[i] - max) / sum;
};



inline double NN::Softmax::fun_prime(std::span<const double> layer, size_t i, size_t j) const noexcept {
  double pi = fun(layer, i);
  double pj = fun(layer, j);

  if(i == j) return pi * (1.0 - pi);

  return -pi * pj;
};



// ====================== //
// ======== ReLU ======== //
// ====================== //
inline double NN::ReLU::fun(std::span<const double> layer, size_t i) const noexcept {
  return std::max(0.0, layer[i]);
};



inline double NN::ReLU::fun_prime(std::span<const double> layer, size_t i, size_t j) const noexcept {
  if(i != j) return 0.0;
  return layer[i] > 0.0 ? 1.0 : 0.0;
};
