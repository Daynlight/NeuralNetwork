// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <math.h>



namespace NN{
class iActivation{
public:
  virtual double fun(double x) const noexcept = 0;
  virtual double fun_prime(double x) const noexcept = 0;
};



class Linear : public iActivation{
public:
  double fun(double x) const noexcept;
  double fun_prime(double x) const noexcept;
};

class Sigmoid : public iActivation{
public:
  double fun(double x) const noexcept;
  double fun_prime(double x) const noexcept;
};
};



#include "Activation.hpp"
