// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Activation.h"



// ===================== //
// ======= Linear ====== //
// ===================== //
inline double NN::Linear::fun(double x) const noexcept {
  return x;
};



inline double NN::Linear::fun_prime(double x) const noexcept {
  return 1; 
};



// ====================== //
// ======= Sigmoid ====== //
// ====================== //
inline double NN::Sigmoid::fun(double x) const noexcept {
  if(x >= 0) return 1.0 / (1.0 + std::exp(-x));
  else {
    double z = std::exp(x);
    return z / (1.0 + z);
  };
};



inline double NN::Sigmoid::fun_prime(double x) const noexcept {
  double y = fun(x);
  return y * (1 - y); 
};
