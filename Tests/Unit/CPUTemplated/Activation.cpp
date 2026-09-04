// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <cmath>
#include <memory>

#include "NeuralNetwork/Activation/Activation.h"



// =============================
// ========= Linear ============
// =============================
TEST(LinearFun, ReturnsInput){
  NN::Linear linear;

  EXPECT_DOUBLE_EQ(linear.fun(0.0), 0.0);
  EXPECT_DOUBLE_EQ(linear.fun(1.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun(-1.0), -1.0);
  EXPECT_DOUBLE_EQ(linear.fun(123.456), 123.456);
  EXPECT_DOUBLE_EQ(linear.fun(-123.456), -123.456);
};

TEST(LinearFun, HandlesExtremeValues){
  NN::Linear linear;

  EXPECT_DOUBLE_EQ(linear.fun(std::numeric_limits<double>::max()), std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(linear.fun(std::numeric_limits<double>::lowest()), std::numeric_limits<double>::lowest());
};

TEST(LinearFunPrime, AlwaysReturnsOne){
  NN::Linear linear;

  EXPECT_DOUBLE_EQ(linear.fun_prime(0.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(1.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(-1.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(123.456), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(-123.456), 1.0);
};

TEST(LinearFunPrime, HandlesExtremeValues){
  NN::Linear linear;

  EXPECT_DOUBLE_EQ(linear.fun_prime(std::numeric_limits<double>::max()), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(std::numeric_limits<double>::lowest()), 1.0);
};



// =============================
// ========= Sigmoid ===========
// =============================
TEST(SigmoidFun, ReturnsExpectedValues){
  NN::Sigmoid sigmoid;

  EXPECT_DOUBLE_EQ(sigmoid.fun(0.0), 0.5);
  EXPECT_NEAR(sigmoid.fun(1.0), 0.7310585786300049, 1e-12);
  EXPECT_NEAR(sigmoid.fun(-1.0), 0.2689414213699951, 1e-12);
};

TEST(SigmoidFun, HandlesLargePositiveValue){
  NN::Sigmoid sigmoid;

  const double result = sigmoid.fun(1000.0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_DOUBLE_EQ(result, 1.0);
};

TEST(SigmoidFun, HandlesLargeNegativeValue){
  NN::Sigmoid sigmoid;

  const double result = sigmoid.fun(-1000.0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_DOUBLE_EQ(result, 0.0);
};

TEST(SigmoidFun, ResultRemainsInRange){
  NN::Sigmoid sigmoid;

  for(double value = -100.0; value <= 100.0; value += 0.5){
    const double result = sigmoid.fun(value);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
  };
};

TEST(SigmoidFun, IsMonotonicallyIncreasing){
  NN::Sigmoid sigmoid;

  double previous = sigmoid.fun(-10.0);

  for(double value = -9.9; value <= 10.0; value += 0.1){
    const double current = sigmoid.fun(value);

    EXPECT_GE(current, previous);

    previous = current;
  };
};

TEST(SigmoidFun, IsSymmetricAroundHalf){
  NN::Sigmoid sigmoid;

  EXPECT_NEAR(sigmoid.fun(1.0) + sigmoid.fun(-1.0), 1.0, 1e-12);
  EXPECT_NEAR(sigmoid.fun(5.0) + sigmoid.fun(-5.0), 1.0, 1e-12);
  EXPECT_NEAR(sigmoid.fun(10.0) + sigmoid.fun(-10.0), 1.0, 1e-12);
};

TEST(SigmoidFun, HandlesInfinity){
  NN::Sigmoid sigmoid;

  EXPECT_DOUBLE_EQ(sigmoid.fun(std::numeric_limits<double>::infinity()), 1.0);
  EXPECT_DOUBLE_EQ(sigmoid.fun(-std::numeric_limits<double>::infinity()), 0.0);
};

TEST(SigmoidFunPrime, ReturnsExpectedValues){
  NN::Sigmoid sigmoid;

  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(0.0), 0.25);
  EXPECT_NEAR(sigmoid.fun_prime(1.0), 0.19661193324148185, 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(-1.0), 0.19661193324148185, 1e-12);
};

TEST(SigmoidFunPrime, ApproachesZeroForExtremeValues){
  NN::Sigmoid sigmoid;

  EXPECT_NEAR(sigmoid.fun_prime(1000.0), 0.0, 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(-1000.0), 0.0, 1e-12);
};

TEST(SigmoidFunPrime, IsSymmetric){
  NN::Sigmoid sigmoid;

  EXPECT_NEAR(sigmoid.fun_prime(1.0), sigmoid.fun_prime(-1.0), 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(5.0), sigmoid.fun_prime(-5.0), 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(10.0), sigmoid.fun_prime(-10.0), 1e-12);
};

TEST(SigmoidFunPrime, MaximumAtZero){
  NN::Sigmoid sigmoid;

  const double zero = sigmoid.fun_prime(0.0);

  for(double value = -10.0; value <= 10.0; value += 0.1)
    EXPECT_LE(sigmoid.fun_prime(value), zero);
};

TEST(SigmoidFunPrime, ResultRemainsInRange){
  NN::Sigmoid sigmoid;

  for(double value = -100.0; value <= 100.0; value += 0.5){
    const double result = sigmoid.fun_prime(value);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 0.25);
  };
};

TEST(SigmoidFunPrime, HandlesInfinity){
  NN::Sigmoid sigmoid;

  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(std::numeric_limits<double>::infinity()), 0.0);
  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(-std::numeric_limits<double>::infinity()), 0.0);
};



// =============================
// ======= Interface ===========
// =============================
TEST(ActivationInterface, LinearWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Linear>();

  EXPECT_DOUBLE_EQ(activation->fun(2.5), 2.5);
  EXPECT_DOUBLE_EQ(activation->fun_prime(2.5), 1.0);
};

TEST(ActivationInterface, SigmoidWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Sigmoid>();

  EXPECT_NEAR(activation->fun(1.0), 0.7310585786300049, 1e-12);
  EXPECT_NEAR(activation->fun_prime(1.0), 0.19661193324148185, 1e-12);
};