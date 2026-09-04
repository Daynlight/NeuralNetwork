// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <cmath>

#define private public
#define protected public

#include "NeuralNetwork/Activation/Activation.h"

#undef private
#undef protected



// =============================
// ========= Linear ============
// =============================
TEST(LinearGetType, ReturnsLinearType) {
  NN::Linear linear;

  EXPECT_EQ(linear.getType(), NN::ActivationType::LINEARTYPE);
};

TEST(LinearFun, ReturnsInput) {
  NN::Linear linear;

  EXPECT_DOUBLE_EQ(linear.fun(0.0), 0.0);
  EXPECT_DOUBLE_EQ(linear.fun(1.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun(-1.0), -1.0);
  EXPECT_DOUBLE_EQ(linear.fun(123.456), 123.456);
};

TEST(LinearFunPrime, AlwaysReturnsOne) {
  NN::Linear linear;

  EXPECT_DOUBLE_EQ(linear.fun_prime(0.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(1.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(-1.0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(1000000.0), 1.0);
};



// =============================
// ========= Sigmoid ===========
// =============================
TEST(SigmoidGetType, ReturnsSigmoidType) {
  NN::Sigmoid sigmoid;

  EXPECT_EQ(sigmoid.getType(), NN::ActivationType::SIGMOIDTYPE);
};

TEST(SigmoidFun, ReturnsExpectedValues) {
  NN::Sigmoid sigmoid;

  EXPECT_DOUBLE_EQ(sigmoid.fun(0.0), 0.5);
  EXPECT_NEAR(sigmoid.fun(1.0), 0.7310585786300049, 1e-12);
  EXPECT_NEAR(sigmoid.fun(-1.0), 0.2689414213699951, 1e-12);
};

TEST(SigmoidFun, HandlesLargePositiveValue) {
  NN::Sigmoid sigmoid;

  const double result = sigmoid.fun(1000.0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_DOUBLE_EQ(result, 1.0);
};

TEST(SigmoidFun, HandlesLargeNegativeValue) {
  NN::Sigmoid sigmoid;

  const double result = sigmoid.fun(-1000.0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_DOUBLE_EQ(result, 0.0);
};

TEST(SigmoidFun, ResultRemainsInRange) {
  NN::Sigmoid sigmoid;

  for(double value = -100.0; value <= 100.0; value += 0.5) {
    const double result = sigmoid.fun(value);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
  };
};

TEST(SigmoidFun, IsMonotonicallyIncreasing) {
  NN::Sigmoid sigmoid;

  double previous = sigmoid.fun(-10.0);

  for(double value = -9.9; value <= 10.0; value += 0.1) {
    const double current = sigmoid.fun(value);

    EXPECT_GE(current, previous);

    previous = current;
  };
};

TEST(SigmoidFunPrime, ReturnsExpectedValues) {
  NN::Sigmoid sigmoid;

  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(0.0), 0.25);
  EXPECT_NEAR(sigmoid.fun_prime(1.0), 0.19661193324148185, 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(-1.0), 0.19661193324148185, 1e-12);
};

TEST(SigmoidFunPrime, ApproachesZeroForExtremeValues) {
  NN::Sigmoid sigmoid;

  EXPECT_NEAR(sigmoid.fun_prime(1000.0), 0.0, 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(-1000.0), 0.0, 1e-12);
};



// =============================
// ========= Factory ===========
// =============================
TEST(ActivationFactory, CreatesLinear) {
  NN::iActivation* activation = NN::getActivationFromType(NN::ActivationType::LINEARTYPE);

  ASSERT_NE(activation, nullptr);
  EXPECT_EQ(activation->getType(), NN::ActivationType::LINEARTYPE);

  delete activation;
};

TEST(ActivationFactory, CreatesSigmoid) {
  NN::iActivation* activation = NN::getActivationFromType(NN::ActivationType::SIGMOIDTYPE);

  ASSERT_NE(activation, nullptr);
  EXPECT_EQ(activation->getType(), NN::ActivationType::SIGMOIDTYPE);

  delete activation;
};

TEST(ActivationFactory, ReturnsNullptrForInvalidType) {
  NN::iActivation* activation = NN::getActivationFromType(static_cast<NN::ActivationType>(100));

  EXPECT_EQ(activation, nullptr);
};