#pragma once


namespace NN{
class iLoss{
public:
  virtual double double fun(double double x, double double t) = 0;
  virtual double double fun_prime(double double x, double double t) = 0;
};

class MSE : public iLoss{
public:
  double double fun(double double x, double double t) { return (x - t)*(x - t)/2; };
  double double fun_prime(double double x, double double t) { return x - t; };
};

};