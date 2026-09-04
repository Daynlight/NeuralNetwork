// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>
#include <cmath>

#define private public
#define protected public

#include "NeuralNetwork/NeuralNetwork.h"

#undef private
#undef protected



// =============================
// ======== LayerBuilder ========
// =============================
TEST(LayerBuilder, BuildsSingleLayer) {
  using Tuple = NN::LayerBuilder<2, 1>::type;

  EXPECT_EQ(std::tuple_size_v<Tuple>, 1u);
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<0, Tuple>, NN::Layer<2, 1>>));
};

TEST(LayerBuilder, BuildsTwoLayers) {
  using Tuple = NN::LayerBuilder<2, 3, 1>::type;

  EXPECT_EQ(std::tuple_size_v<Tuple>, 2u);
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<0, Tuple>, NN::Layer<2, 3>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<1, Tuple>, NN::Layer<3, 1>>));
};

TEST(LayerBuilder, BuildsMultipleLayers) {
  using Tuple = NN::LayerBuilder<3, 8, 4, 2, 1>::type;

  EXPECT_EQ(std::tuple_size_v<Tuple>, 4u);
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<0, Tuple>, NN::Layer<3, 8>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<1, Tuple>, NN::Layer<8, 4>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<2, Tuple>, NN::Layer<4, 2>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<3, Tuple>, NN::Layer<2, 1>>));
};



// =============================
// ===== Reverse Sequence ======
// =============================
TEST(ReverseSequence, ReversesSingleElement) {
  using Result = decltype(NN::reverse_sequence(std::make_index_sequence<1>{}));
  using Expected = std::index_sequence<0>;

  EXPECT_TRUE((std::is_same_v<Result, Expected>));
};

TEST(ReverseSequence, ReversesTwoElements) {
  using Result = decltype(NN::reverse_sequence(std::make_index_sequence<2>{}));
  using Expected = std::index_sequence<1, 0>;

  EXPECT_TRUE((std::is_same_v<Result, Expected>));
};

TEST(ReverseSequence, ReversesMultipleElements) {
  using Result = decltype(NN::reverse_sequence(std::make_index_sequence<5>{}));
  using Expected = std::index_sequence<4, 3, 2, 1, 0>;

  EXPECT_TRUE((std::is_same_v<Result, Expected>));
};



// =============================
// ======= Constructors ========
// =============================
TEST(NeuralNetworkDefaultConstructor, InitializesLayers) {
  NN::NeuralNetwork<2, 3, 1> network;

  EXPECT_EQ(std::tuple_size_v<decltype(network.layers)>, 2u);
};

TEST(NeuralNetworkDefaultConstructor, InitializesDefaultLearningRate) {
  NN::NeuralNetwork<2, 3, 1> network;

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers).getLearningRate(), 0.005);
  EXPECT_DOUBLE_EQ(std::get<1>(network.layers).getLearningRate(), 0.005);
};

TEST(NeuralNetworkDefaultConstructor, InitializesDefaultActivation) {
  NN::NeuralNetwork<2, 3, 1> network;

  NN::iActivation* first = const_cast<NN::iActivation*>(std::get<0>(network.layers).getActivation());
  NN::iActivation* second = const_cast<NN::iActivation*>(std::get<1>(network.layers).getActivation());

  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  EXPECT_EQ(first->getType(), NN::ActivationType::LINEARTYPE);
  EXPECT_EQ(second->getType(), NN::ActivationType::LINEARTYPE);
};

TEST(NeuralNetworkDefaultConstructor, InitializesDefaultLoss) {
  NN::NeuralNetwork<2, 3, 1> network;

  NN::iLoss* first = const_cast<NN::iLoss*>(std::get<0>(network.layers).getLoss());
  NN::iLoss* second = const_cast<NN::iLoss*>(std::get<1>(network.layers).getLoss());

  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  EXPECT_EQ(first->getType(), NN::LossType::MSETYPE);
  EXPECT_EQ(second->getType(), NN::LossType::MSETYPE);
};



// =============================
// ===== Learning Rate =========
// =============================
TEST(NeuralNetworkSetLearningRate, SetsSingleLayer) {
  NN::NeuralNetwork<2, 1> network;

  network.setLearningRate(0.001);

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers).getLearningRate(), 0.001);
};

TEST(NeuralNetworkSetLearningRate, SetsEveryLayer) {
  NN::NeuralNetwork<2, 3, 2, 1> network;

  network.setLearningRate(0.001);

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers).getLearningRate(), 0.001);
  EXPECT_DOUBLE_EQ(std::get<1>(network.layers).getLearningRate(), 0.001);
  EXPECT_DOUBLE_EQ(std::get<2>(network.layers).getLearningRate(), 0.001);
};

TEST(NeuralNetworkSetLearningRate, ReplacesPreviousLearningRate) {
  NN::NeuralNetwork<2, 3, 1> network;

  network.setLearningRate(0.1);
  network.setLearningRate(0.0001);

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers).getLearningRate(), 0.0001);
  EXPECT_DOUBLE_EQ(std::get<1>(network.layers).getLearningRate(), 0.0001);
};



// =============================
// ======== Activation =========
// =============================
TEST(NeuralNetworkSetActivation, SetsFirstLayer) {
  NN::NeuralNetwork<2, 3, 1> network;

  network.setActivation<0>(NN::ActivationType::SIGMOIDTYPE);

  NN::iActivation* activation = const_cast<NN::iActivation*>(std::get<0>(network.layers).getActivation());

  ASSERT_NE(activation, nullptr);
  EXPECT_EQ(activation->getType(), NN::ActivationType::SIGMOIDTYPE);
};

TEST(NeuralNetworkSetActivation, SetsRequestedLayerOnly) {
  NN::NeuralNetwork<2, 3, 2, 1> network;

  network.setActivation<1>(NN::ActivationType::SIGMOIDTYPE);

  NN::iActivation* first = const_cast<NN::iActivation*>(std::get<0>(network.layers).getActivation());
  NN::iActivation* second = const_cast<NN::iActivation*>(std::get<1>(network.layers).getActivation());
  NN::iActivation* third = const_cast<NN::iActivation*>(std::get<2>(network.layers).getActivation());

  EXPECT_EQ(first->getType(), NN::ActivationType::LINEARTYPE);
  EXPECT_EQ(second->getType(), NN::ActivationType::SIGMOIDTYPE);
  EXPECT_EQ(third->getType(), NN::ActivationType::LINEARTYPE);
};

TEST(NeuralNetworkSetActivation, ReplacesPreviousActivation) {
  NN::NeuralNetwork<2, 3, 1> network;

  network.setActivation<0>(NN::ActivationType::SIGMOIDTYPE);
  network.setActivation<0>(NN::ActivationType::LINEARTYPE);

  NN::iActivation* activation = const_cast<NN::iActivation*>(std::get<0>(network.layers).getActivation());

  ASSERT_NE(activation, nullptr);
  EXPECT_EQ(activation->getType(), NN::ActivationType::LINEARTYPE);
};



// =============================
// ======== Node Control ========
// =============================
TEST(NeuralNetworksetInput, SetsInputLayerNodes) {
  NN::NeuralNetwork<3, 2, 1> network;

  network.setInput({1.0, 2.0, 3.0});

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[0], 1.0);
  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[1], 2.0);
  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[2], 3.0);
};

TEST(NeuralNetworksetInput, DoesNotModifyOtherLayers) {
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<1>(network.layers).setNodes({8.0, 9.0});

  network.setInput({1.0, 2.0});

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[0], 1.0);
  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[1], 2.0);

  EXPECT_DOUBLE_EQ(std::get<1>(network.layers)[0], 8.0);
  EXPECT_DOUBLE_EQ(std::get<1>(network.layers)[1], 9.0);
};

TEST(NeuralNetworksetInput, IgnoresAdditionalNodes) {
  NN::NeuralNetwork<2, 2, 1> network;

  network.setInput({1.0, 2.0, 3.0, 4.0});

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[0], 1.0);
  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[1], 2.0);
};



// =============================
// ========== Forward ==========
// =============================
TEST(NeuralNetworkForward, ForwardsSingleConnection) {
  NN::NeuralNetwork<2, 1> network;

  network.setInput({2.0, 3.0});
  std::get<0>(network.layers).setWeights({4.0, 5.0, 6.0});

  network.forward();

  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[0], 2.0);
  EXPECT_DOUBLE_EQ(std::get<0>(network.layers)[1], 3.0);
};

TEST(NeuralNetworkForward, ForwardsThroughTwoLayers) {
  NN::NeuralNetwork<2, 2, 1> network;

  network.setInput({2.0, 3.0});

  std::get<0>(network.layers).setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  network.forward();

  EXPECT_DOUBLE_EQ(std::get<1>(network.layers)[0], 11.0);
  EXPECT_DOUBLE_EQ(std::get<1>(network.layers)[1], 29.0);
};

TEST(NeuralNetworkForward, ForwardsThroughMultipleLayers) {
  NN::NeuralNetwork<2, 2, 2, 1> network;

  network.setInput({1.0, 2.0});

  std::get<0>(network.layers).setWeights({
    1.0, 1.0, 1.0,
    2.0, 1.0, 0.0
  });

  std::get<1>(network.layers).setWeights({
    1.0, 2.0, 1.0,
    3.0, 1.0, 2.0
  });

  network.forward();

  EXPECT_DOUBLE_EQ(std::get<1>(network.layers)[0], 4.0);
  EXPECT_DOUBLE_EQ(std::get<1>(network.layers)[1], 4.0);

  EXPECT_DOUBLE_EQ(std::get<2>(network.layers)[0], 13.0);
  EXPECT_DOUBLE_EQ(std::get<2>(network.layers)[1], 18.0);
};

TEST(NeuralNetworkForward, AppliesActivation) {
  NN::NeuralNetwork<1, 1, 1> network;

  network.setInput({0.0});

  std::get<0>(network.layers).setWeights({2.0, 0.0});
  network.setActivation<0>(NN::ActivationType::SIGMOIDTYPE);

  network.forward();

  EXPECT_DOUBLE_EQ(std::get<1>(network.layers)[0], 1.0);
};

TEST(NeuralNetworkForward, ProducesFiniteValues) {
  NN::NeuralNetwork<3, 8, 8, 1> network;

  network.setInput({0.2, 0.4, 0.6});

  network.forward();

  double* result = network.getResult();

  ASSERT_NE(result, nullptr);

  for(unsigned int i = 0; i < 8; ++i)
    EXPECT_TRUE(std::isfinite(result[i]));
};



// =============================
// ========= Backprop ==========
// =============================
TEST(NeuralNetworkBackpropInitial, UpdatesLastConnectionWeights) {
  NN::NeuralNetwork<1, 1, 1> network;

  network.setLearningRate(0.1);
  network.setInput({2.0});

  std::get<0>(network.layers).setWeights({1.0, 0.0});
  std::get<1>(network.layers).setWeights({1.0, 0.0});

  network.forward();

  const double before = std::get<0>(network.layers).getWeights()[0];

  network.backprop({1.0});

  EXPECT_NE(std::get<0>(network.layers).getWeights()[0], before);
};

TEST(NeuralNetworkBackprop, UpdatesWeights) {
  NN::NeuralNetwork<2, 2, 1> network;

  network.setLearningRate(0.001);
  network.setInput({0.5, 0.25});
  network.forward();

  const double first_before = std::get<0>(network.layers).getWeights()[0];

  network.backprop({0.5});

  EXPECT_NE(std::get<0>(network.layers).getWeights()[0], first_before);
};

TEST(NeuralNetworkBackprop, UpdatesMultipleLayers) {
  NN::NeuralNetwork<2, 2, 2, 1> network;

  network.setLearningRate(0.001);
  network.setInput({0.25, 0.5});
  network.forward();

  const double first_before = std::get<0>(network.layers).getWeights()[0];
  const double second_before = std::get<1>(network.layers).getWeights()[0];

  network.backprop({0.5, 0.5});

  EXPECT_NE(std::get<0>(network.layers).getWeights()[0], first_before);
  EXPECT_NE(std::get<1>(network.layers).getWeights()[0], second_before);
};

TEST(NeuralNetworkBackprop, KeepsWeightsFinite) {
  NN::NeuralNetwork<2, 4, 4, 1> network;

  network.setLearningRate(0.0001);

  for(unsigned int i = 0; i < 100; ++i) {
    network.setInput({0.25, 0.5});
    network.forward();
    network.backprop({0.5, 0.5, 0.5, 0.5});
  };

  for(double weight : std::get<0>(network.layers).weights)
    EXPECT_TRUE(std::isfinite(weight));

  for(double weight : std::get<1>(network.layers).weights)
    EXPECT_TRUE(std::isfinite(weight));

  for(double weight : std::get<2>(network.layers).weights)
    EXPECT_TRUE(std::isfinite(weight));
};

TEST(NeuralNetworkBackprop, KeepsSigmaFinite) {
  NN::NeuralNetwork<2, 3, 3, 1> network;

  network.setLearningRate(0.0001);
  network.setInput({0.2, 0.4});
  network.forward();
  network.backprop({0.5, 0.5, 0.5});

  for(double sigma : std::get<0>(network.layers).sigma)
    EXPECT_TRUE(std::isfinite(sigma));

  for(double sigma : std::get<1>(network.layers).sigma)
    EXPECT_TRUE(std::isfinite(sigma));

  for(double sigma : std::get<2>(network.layers).sigma)
    EXPECT_TRUE(std::isfinite(sigma));
};



// =============================
// =========== Result ==========
// =============================
TEST(NeuralNetworkGetResult, ReturnsLastLayerNodes) {
  NN::NeuralNetwork<2, 2, 1> network;

  EXPECT_EQ(network.getResult(), std::get<1>(network.layers).getNodes());
};

TEST(NeuralNetworkGetResult, ReflectsLastLayerValues) {
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<1>(network.layers).setNodes({4.0, 8.0});

  double* result = network.getResult();

  ASSERT_NE(result, nullptr);
  EXPECT_DOUBLE_EQ(result[0], 4.0);
  EXPECT_DOUBLE_EQ(result[1], 8.0);
};

TEST(NeuralNetworkGetResult, ReflectsForwardResult) {
  NN::NeuralNetwork<2, 2, 1> network;

  network.setInput({2.0, 3.0});

  std::get<0>(network.layers).setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  network.forward();

  double* result = network.getResult();

  ASSERT_NE(result, nullptr);
  EXPECT_DOUBLE_EQ(result[0], 11.0);
  EXPECT_DOUBLE_EQ(result[1], 29.0);
};

TEST(NeuralNetworkGetResult, ReturnsFiniteValues) {
  NN::NeuralNetwork<3, 8, 8, 1> network;

  network.setInput({0.2, 0.4, 0.6});
  network.forward();

  double* result = network.getResult();

  ASSERT_NE(result, nullptr);

  for(unsigned int i = 0; i < 8; ++i)
    EXPECT_TRUE(std::isfinite(result[i]));
};



// =============================
// ======= Integration =========
// =============================
TEST(NeuralNetworkIntegration, ForwardBackpropForwardChangesResult) {
  NN::NeuralNetwork<2, 2, 1> network;

  network.setLearningRate(0.001);
  network.setInput({0.5, 0.25});
  network.forward();

  const double before = network.getResult()[0];

  network.backprop({0.5, 0.5});

  network.setInput({0.5, 0.25});
  network.forward();

  const double after = network.getResult()[0];

  EXPECT_NE(before, after);
};

TEST(NeuralNetworkIntegration, RepeatedTrainingKeepsResultFinite) {
  NN::NeuralNetwork<2, 4, 4, 1> network;

  network.setLearningRate(0.0001);

  for(unsigned int i = 0; i < 1000; ++i) {
    network.setInput({0.2, 0.4});
    network.forward();
    network.backprop({0.5, 0.5, 0.5, 0.5});
  };

  network.setInput({0.2, 0.4});
  network.forward();

  double* result = network.getResult();

  ASSERT_NE(result, nullptr);

  for(unsigned int i = 0; i < 4; ++i)
    EXPECT_TRUE(std::isfinite(result[i]));
};

TEST(NeuralNetworkIntegration, DifferentInputsProduceDifferentResults) {
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<0>(network.layers).setWeights({
    1.0, 2.0, 0.0,
    3.0, 4.0, 0.0
  });

  network.setInput({1.0, 2.0});
  network.forward();

  const double first_a = network.getResult()[0];
  const double first_b = network.getResult()[1];

  network.setInput({3.0, 4.0});
  network.forward();

  const double second_a = network.getResult()[0];
  const double second_b = network.getResult()[1];

  EXPECT_NE(first_a, second_a);
  EXPECT_NE(first_b, second_b);
};