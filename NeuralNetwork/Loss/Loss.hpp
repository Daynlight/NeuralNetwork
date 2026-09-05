// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Loss.h"



// ====================== //
// ========= MSE ======== //
// ====================== //
inline double NN::MSE::fun(double x, double t) const noexcept {
  return (x - t)*(x - t)/2;
};



inline double NN::MSE::fun_prime(double x, double t) const noexcept {
  return x - t;
};



// ====================== //
// ==== CrossEntropy ==== //
// ====================== //
inline double NN::CrossEntropy::fun(double x, double t) const noexcept {
  constexpr double epsilon = 1e-12;
  x = std::clamp(x, epsilon, 1.0);
  return -t * std::log(x);
};



inline double NN::CrossEntropy::fun_prime(double x, double t) const noexcept {
  constexpr double epsilon = 1e-12;
  x = std::clamp(x, epsilon, 1.0);
  return -t / x;
};