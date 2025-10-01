#pragma once
#include <math.h>

namespace NN{
enum ActivationType {
  LINEARTYPE = 0,
  SIGMOIDTYPE = 1
};

class iActivation{
public:
  virtual ActivationType getType() = 0;
  virtual double fun(double x) = 0;
  virtual double fun_prime(double x) = 0;
};

class Linear : public iActivation{
  ActivationType getType() { return ActivationType::LINEARTYPE; };
  double fun(double x) { return x; };
  double fun_prime(double x) { return 1; };
};

class Sigmoid : public iActivation{
  ActivationType getType() { return ActivationType::SIGMOIDTYPE; };
  double fun(double x) { return 1 / (1 + exp(-x)); }
  double fun_prime(double x) { return fun(x) * (1 - fun(x)); };
};

iActivation* getActivationFromType(ActivationType type){
  if(type == ActivationType::LINEARTYPE) return new Linear();
  if(type == ActivationType::SIGMOIDTYPE) return new Sigmoid();
  return nullptr;
};
};