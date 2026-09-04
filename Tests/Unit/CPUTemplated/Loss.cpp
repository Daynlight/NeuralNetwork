// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>

#include "NeuralNetwork/Loss/Loss.h"



// =============================
// =========== MSE =============
// =============================
TEST(MSEFun, ReturnsZeroForEqualValues){
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun(1.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun(-1.0, -1.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun(123.456, 123.456), 0.0);
};

TEST(MSEFun, ReturnsExpectedLoss){
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun(2.0, 1.0), 0.5);
  EXPECT_DOUBLE_EQ(mse.fun(1.0, 2.0), 0.5);
  EXPECT_DOUBLE_EQ(mse.fun(5.0, 1.0), 8.0);
  EXPECT_DOUBLE_EQ(mse.fun(-2.0, 2.0), 8.0);
  EXPECT_DOUBLE_EQ(mse.fun(-5.0, -1.0), 8.0);
};

TEST(MSEFun, IsSymmetric){
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun(5.0, 2.0), mse.fun(2.0, 5.0));
  EXPECT_DOUBLE_EQ(mse.fun(-5.0, 2.0), mse.fun(2.0, -5.0));
  EXPECT_DOUBLE_EQ(mse.fun(-5.0, -2.0), mse.fun(-2.0, -5.0));
};

TEST(MSEFun, IsAlwaysNonNegative){
  NN::MSE mse;

  for(double x = -10.0; x <= 10.0; x += 0.5)
    for(double t = -10.0; t <= 10.0; t += 0.5)
      EXPECT_GE(mse.fun(x, t), 0.0);
};

TEST(MSEFun, IncreasesWithDistance){
  NN::MSE mse;

  const double target = 0.0;

  EXPECT_LT(mse.fun(1.0, target), mse.fun(2.0, target));
  EXPECT_LT(mse.fun(2.0, target), mse.fun(3.0, target));
  EXPECT_LT(mse.fun(-1.0, target), mse.fun(-2.0, target));
  EXPECT_LT(mse.fun(-2.0, target), mse.fun(-3.0, target));
};

TEST(MSEFun, HandlesLargeFiniteValues){
  NN::MSE mse;

  const double result = mse.fun(1e100, 0.0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_DOUBLE_EQ(result, 5e199);
};

TEST(MSEFunPrime, ReturnsZeroForEqualValues){
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun_prime(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(5.0, 5.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(-5.0, -5.0), 0.0);
};

TEST(MSEFunPrime, ReturnsExpectedDerivative){
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun_prime(5.0, 2.0), 3.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(2.0, 5.0), -3.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(-2.0, 2.0), -4.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(2.0, -2.0), 4.0);
};

TEST(MSEFunPrime, ChangesSignWhenArgumentsAreSwapped){
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun_prime(5.0, 2.0), -mse.fun_prime(2.0, 5.0));
  EXPECT_DOUBLE_EQ(mse.fun_prime(-5.0, 2.0), -mse.fun_prime(2.0, -5.0));
  EXPECT_DOUBLE_EQ(mse.fun_prime(-5.0, -2.0), -mse.fun_prime(-2.0, -5.0));
};

TEST(MSEFunPrime, SignPointsAwayFromTarget){
  NN::MSE mse;

  EXPECT_GT(mse.fun_prime(5.0, 2.0), 0.0);
  EXPECT_LT(mse.fun_prime(2.0, 5.0), 0.0);
  EXPECT_GT(mse.fun_prime(1.0, 0.0), 0.0);
  EXPECT_LT(mse.fun_prime(-1.0, 0.0), 0.0);
};

TEST(MSEFunPrime, HandlesLargeFiniteValues){
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun_prime(1e100, 0.0), 1e100);
  EXPECT_DOUBLE_EQ(mse.fun_prime(-1e100, 0.0), -1e100);
};



// =============================
// ======== Interface ==========
// =============================
TEST(LossInterface, MSEWorksPolymorphically){
  std::unique_ptr<NN::iLoss> loss = std::make_unique<NN::MSE>();

  EXPECT_DOUBLE_EQ(loss->fun(5.0, 2.0), 4.5);
  EXPECT_DOUBLE_EQ(loss->fun_prime(5.0, 2.0), 3.0);
};