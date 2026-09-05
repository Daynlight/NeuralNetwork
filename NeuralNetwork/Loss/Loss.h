// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <algorithm>
#include <cmath>



namespace NN{
class iLoss{
public:
  virtual double fun(double x, double t) const noexcept = 0;
  virtual double fun_prime(double x, double t) const noexcept = 0;
};



class MSE : public iLoss{
public:
  double fun(double x, double t) const noexcept;
  double fun_prime(double x, double t) const noexcept;
};

class CrossEntropy : public iLoss{
public:
  double fun(double x, double t) const noexcept;
  double fun_prime(double x, double t) const noexcept;
};
};



#include "Loss.hpp"
