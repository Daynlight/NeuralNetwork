#pragma once


namespace NN{
enum LossType{
  MSETYPE = 0
};

class iLoss{
public:
  virtual LossType getType() = 0;
  virtual double fun(double x, double t) = 0;
  virtual double fun_prime(double x, double t) = 0;
};

class MSE : public iLoss{
public:
  LossType getType() { return LossType::MSETYPE; };
  double fun(double x, double t) { return (x - t)*(x - t)/2; };
  double fun_prime(double x, double t) { return x - t; };
};

iLoss *getLossFromType(LossType type) {
  if(type == MSETYPE) return new MSE();
  return nullptr;
};
};