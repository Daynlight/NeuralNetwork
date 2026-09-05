// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "NeuralNetwork/Activation/Activation.h"



// =============================
// ========= Linear ============
// =============================
TEST(LinearFun, ReturnsInput){
  NN::Linear linear;

  std::array<double, 5> layer = {0.0, 1.0, -1.0, 123.456, -123.456};

  EXPECT_DOUBLE_EQ(linear.fun(layer, 0), 0.0);
  EXPECT_DOUBLE_EQ(linear.fun(layer, 1), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun(layer, 2), -1.0);
  EXPECT_DOUBLE_EQ(linear.fun(layer, 3), 123.456);
  EXPECT_DOUBLE_EQ(linear.fun(layer, 4), -123.456);
};

TEST(LinearFun, HandlesExtremeValues){
  NN::Linear linear;

  std::array<double, 2> layer = {
    std::numeric_limits<double>::max(),
    std::numeric_limits<double>::lowest()
  };

  EXPECT_DOUBLE_EQ(linear.fun(layer, 0), std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(linear.fun(layer, 1), std::numeric_limits<double>::lowest());
};

TEST(LinearFunPrime, ReturnsIdentityJacobian){
  NN::Linear linear;

  std::array<double, 3> layer = {1.0, 2.0, 3.0};

  EXPECT_DOUBLE_EQ(linear.fun_prime(layer, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(layer, 1, 1), 1.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(layer, 2, 2), 1.0);

  EXPECT_DOUBLE_EQ(linear.fun_prime(layer, 0, 1), 0.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(layer, 0, 2), 0.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(layer, 1, 0), 0.0);
  EXPECT_DOUBLE_EQ(linear.fun_prime(layer, 2, 1), 0.0);
};



// =============================
// ========= Sigmoid ===========
// =============================
TEST(SigmoidFun, ReturnsExpectedValues){
  NN::Sigmoid sigmoid;

  std::array<double, 3> layer = {0.0, 1.0, -1.0};

  EXPECT_DOUBLE_EQ(sigmoid.fun(layer, 0), 0.5);
  EXPECT_NEAR(sigmoid.fun(layer, 1), 0.7310585786300049, 1e-12);
  EXPECT_NEAR(sigmoid.fun(layer, 2), 0.2689414213699951, 1e-12);
};

TEST(SigmoidFun, HandlesLargePositiveValue){
  NN::Sigmoid sigmoid;

  std::array<double, 1> layer = {1000.0};

  const double result = sigmoid.fun(layer, 0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_DOUBLE_EQ(result, 1.0);
};

TEST(SigmoidFun, HandlesLargeNegativeValue){
  NN::Sigmoid sigmoid;

  std::array<double, 1> layer = {-1000.0};

  const double result = sigmoid.fun(layer, 0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_DOUBLE_EQ(result, 0.0);
};

TEST(SigmoidFun, ResultRemainsInRange){
  NN::Sigmoid sigmoid;

  for(double value = -100.0; value <= 100.0; value += 0.5){
    std::array<double, 1> layer = {value};

    const double result = sigmoid.fun(layer, 0);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
  };
};

TEST(SigmoidFun, IsMonotonicallyIncreasing){
  NN::Sigmoid sigmoid;

  std::array<double, 1> layer = {-10.0};

  double previous = sigmoid.fun(layer, 0);

  for(double value = -9.9; value <= 10.0; value += 0.1){
    layer[0] = value;

    const double current = sigmoid.fun(layer, 0);

    EXPECT_GE(current, previous);

    previous = current;
  };
};

TEST(SigmoidFun, IsSymmetricAroundHalf){
  NN::Sigmoid sigmoid;

  std::array<double, 2> layer = {1.0, -1.0};

  EXPECT_NEAR(sigmoid.fun(layer, 0) + sigmoid.fun(layer, 1), 1.0, 1e-12);

  layer = {5.0, -5.0};

  EXPECT_NEAR(sigmoid.fun(layer, 0) + sigmoid.fun(layer, 1), 1.0, 1e-12);

  layer = {10.0, -10.0};

  EXPECT_NEAR(sigmoid.fun(layer, 0) + sigmoid.fun(layer, 1), 1.0, 1e-12);
};

TEST(SigmoidFun, HandlesInfinity){
  NN::Sigmoid sigmoid;

  std::array<double, 2> layer = {
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()
  };

  EXPECT_DOUBLE_EQ(sigmoid.fun(layer, 0), 1.0);
  EXPECT_DOUBLE_EQ(sigmoid.fun(layer, 1), 0.0);
};

TEST(SigmoidFunPrime, ReturnsExpectedValues){
  NN::Sigmoid sigmoid;

  std::array<double, 3> layer = {0.0, 1.0, -1.0};

  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(layer, 0, 0), 0.25);
  EXPECT_NEAR(sigmoid.fun_prime(layer, 1, 1), 0.19661193324148185, 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(layer, 2, 2), 0.19661193324148185, 1e-12);
};

TEST(SigmoidFunPrime, ReturnsZeroOutsideDiagonal){
  NN::Sigmoid sigmoid;

  std::array<double, 3> layer = {0.0, 1.0, -1.0};

  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(layer, 0, 1), 0.0);
  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(layer, 0, 2), 0.0);
  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(layer, 1, 0), 0.0);
  EXPECT_DOUBLE_EQ(sigmoid.fun_prime(layer, 2, 1), 0.0);
};

TEST(SigmoidFunPrime, ApproachesZeroForExtremeValues){
  NN::Sigmoid sigmoid;

  std::array<double, 2> layer = {1000.0, -1000.0};

  EXPECT_NEAR(sigmoid.fun_prime(layer, 0, 0), 0.0, 1e-12);
  EXPECT_NEAR(sigmoid.fun_prime(layer, 1, 1), 0.0, 1e-12);
};

TEST(SigmoidFunPrime, MaximumAtZero){
  NN::Sigmoid sigmoid;

  std::array<double, 1> layer = {0.0};

  const double zero = sigmoid.fun_prime(layer, 0, 0);

  for(double value = -10.0; value <= 10.0; value += 0.1){
    layer[0] = value;

    EXPECT_LE(sigmoid.fun_prime(layer, 0, 0), zero);
  };
};

TEST(SigmoidFunPrime, ResultRemainsInRange){
  NN::Sigmoid sigmoid;

  std::array<double, 1> layer;

  for(double value = -100.0; value <= 100.0; value += 0.5){
    layer[0] = value;

    const double result = sigmoid.fun_prime(layer, 0, 0);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 0.25);
  };
};



// =============================
// ========= Softmax ===========
// =============================
TEST(SoftmaxFun, ProbabilitiesSumToOne){
  NN::Softmax softmax;

  std::array<double, 3> layer = {1.0, 2.0, 3.0};

  double sum = 0.0;

  for(size_t i = 0; i < layer.size(); i++)
    sum += softmax.fun(layer, i);

  EXPECT_NEAR(sum, 1.0, 1e-12);
};

TEST(SoftmaxFun, ReturnsPositiveProbabilities){
  NN::Softmax softmax;

  std::array<double, 4> layer = {-10.0, -1.0, 0.0, 10.0};

  for(size_t i = 0; i < layer.size(); i++){
    EXPECT_GT(softmax.fun(layer, i), 0.0);
    EXPECT_LE(softmax.fun(layer, i), 1.0);
  };
};

TEST(SoftmaxFun, PreservesOrdering){
  NN::Softmax softmax;

  std::array<double, 3> layer = {1.0, 2.0, 3.0};

  EXPECT_LT(softmax.fun(layer, 0), softmax.fun(layer, 1));
  EXPECT_LT(softmax.fun(layer, 1), softmax.fun(layer, 2));
};

TEST(SoftmaxFun, UniformForEqualInputs){
  NN::Softmax softmax;

  std::array<double, 4> layer = {1.0, 1.0, 1.0, 1.0};

  for(size_t i = 0; i < layer.size(); i++)
    EXPECT_NEAR(softmax.fun(layer, i), 0.25, 1e-12);
};

TEST(SoftmaxFun, IsShiftInvariant){
  NN::Softmax softmax;

  std::array<double, 3> first = {1.0, 2.0, 3.0};
  std::array<double, 3> second = {101.0, 102.0, 103.0};

  for(size_t i = 0; i < first.size(); i++)
    EXPECT_NEAR(softmax.fun(first, i), softmax.fun(second, i), 1e-12);
};

TEST(SoftmaxFun, HandlesLargeValues){
  NN::Softmax softmax;

  std::array<double, 3> layer = {1000.0, 1001.0, 1002.0};

  double sum = 0.0;

  for(size_t i = 0; i < layer.size(); i++){
    const double result = softmax.fun(layer, i);

    EXPECT_TRUE(std::isfinite(result));

    sum += result;
  };

  EXPECT_NEAR(sum, 1.0, 1e-12);
};

TEST(SoftmaxFunPrime, DiagonalIsPositive){
  NN::Softmax softmax;

  std::array<double, 3> layer = {1.0, 2.0, 3.0};

  for(size_t i = 0; i < layer.size(); i++)
    EXPECT_GT(softmax.fun_prime(layer, i, i), 0.0);
};

TEST(SoftmaxFunPrime, OffDiagonalIsNegative){
  NN::Softmax softmax;

  std::array<double, 3> layer = {1.0, 2.0, 3.0};

  EXPECT_LT(softmax.fun_prime(layer, 0, 1), 0.0);
  EXPECT_LT(softmax.fun_prime(layer, 0, 2), 0.0);
  EXPECT_LT(softmax.fun_prime(layer, 1, 0), 0.0);
  EXPECT_LT(softmax.fun_prime(layer, 2, 1), 0.0);
};

TEST(SoftmaxFunPrime, JacobianRowsSumToZero){
  NN::Softmax softmax;

  std::array<double, 3> layer = {1.0, 2.0, 3.0};

  for(size_t i = 0; i < layer.size(); i++){
    double sum = 0.0;

    for(size_t j = 0; j < layer.size(); j++)
      sum += softmax.fun_prime(layer, i, j);

    EXPECT_NEAR(sum, 0.0, 1e-12);
  };
};



// =============================
// ======= Interface ===========
// =============================
TEST(ActivationInterface, LinearWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Linear>();

  std::array<double, 2> layer = {2.5, 5.0};

  EXPECT_DOUBLE_EQ(activation->fun(layer, 0), 2.5);
  EXPECT_DOUBLE_EQ(activation->fun_prime(layer, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(activation->fun_prime(layer, 0, 1), 0.0);
};

TEST(ActivationInterface, SigmoidWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Sigmoid>();

  std::array<double, 2> layer = {1.0, -1.0};

  EXPECT_NEAR(activation->fun(layer, 0), 0.7310585786300049, 1e-12);
  EXPECT_NEAR(activation->fun_prime(layer, 0, 0), 0.19661193324148185, 1e-12);
  EXPECT_DOUBLE_EQ(activation->fun_prime(layer, 0, 1), 0.0);
};

TEST(ActivationInterface, SoftmaxWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Softmax>();

  std::array<double, 3> layer = {1.0, 2.0, 3.0};

  double sum = 0.0;

  for(size_t i = 0; i < layer.size(); i++)
    sum += activation->fun(layer, i);

  EXPECT_NEAR(sum, 1.0, 1e-12);
};