#pragma once
#include <math.h>

namespace NN{
enum ActivationType {
  SIGMOIDTYPE = 0
};
class iActivation{
public:
  virtual ActivationType getType() = 0;
  virtual double fun(double x) = 0;
  virtual double fun_prime(double x) = 0;
};

class Sigmoid : public iActivation{
  ActivationType getType() { return ActivationType::SIGMOIDTYPE; };
  double fun(double x) { return 1/(1 + powf64(2.1736, (-1 * x))); };
  double fun_prime(double x) { return fun(x) * (1 - fun(x)); };
};

};