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
  double fun(double x) { 
    if(x >= 0){
      return 1.0 / (1.0 + std::exp(-x));
    }
    else {
      double z = std::exp(x);
      return z / (1.0 + z);
    };
  };
  double fun_prime(double x) { 
    double y = fun(x);
    return y * (1 - y); 
  };
};

iActivation* getActivationFromType(ActivationType type){
  if(type == ActivationType::LINEARTYPE) return new Linear();
  if(type == ActivationType::SIGMOIDTYPE) return new Sigmoid();
  return nullptr;
};
};