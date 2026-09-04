// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>

#include "NeuralNetwork/Loss/Loss.h"



// =============================
// =========== MSE =============
// =============================
TEST(MSEGetType, ReturnsMSEType) {
  NN::MSE mse;

  EXPECT_EQ(mse.getType(), NN::LossType::MSETYPE);
};

TEST(MSEFun, ReturnsZeroForEqualValues) {
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun(1.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun(-5.0, -5.0), 0.0);
};

TEST(MSEFun, ReturnsExpectedLoss) {
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun(2.0, 1.0), 0.5);
  EXPECT_DOUBLE_EQ(mse.fun(1.0, 2.0), 0.5);
  EXPECT_DOUBLE_EQ(mse.fun(5.0, 1.0), 8.0);
  EXPECT_DOUBLE_EQ(mse.fun(-2.0, 2.0), 8.0);
};

TEST(MSEFun, IsSymmetric) {
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun(5.0, 2.0), mse.fun(2.0, 5.0));
  EXPECT_DOUBLE_EQ(mse.fun(-5.0, 2.0), mse.fun(2.0, -5.0));
};

TEST(MSEFunPrime, ReturnsZeroForEqualValues) {
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun_prime(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(5.0, 5.0), 0.0);
};

TEST(MSEFunPrime, ReturnsExpectedDerivative) {
  NN::MSE mse;

  EXPECT_DOUBLE_EQ(mse.fun_prime(5.0, 2.0), 3.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(2.0, 5.0), -3.0);
  EXPECT_DOUBLE_EQ(mse.fun_prime(-2.0, 2.0), -4.0);
};



// =============================
// ========= Factory ===========
// =============================
TEST(LossFactory, CreatesMSE) {
  NN::iLoss* loss = NN::getLossFromType(NN::LossType::MSETYPE);

  ASSERT_NE(loss, nullptr);
  EXPECT_EQ(loss->getType(), NN::LossType::MSETYPE);

  delete loss;
};

TEST(LossFactory, ReturnsNullptrForInvalidType) {
  NN::iLoss* loss = NN::getLossFromType(static_cast<NN::LossType>(100));

  EXPECT_EQ(loss, nullptr);
};