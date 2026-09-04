// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <cmath>
#include <string>

#define private public
#define protected public

#include "NeuralNetwork/Layer/Layer.h"

#undef private
#undef protected



// =============================
// ======= Constructors ========
// =============================
TEST(LayerDefaultConstructor, InitializesWeights) {
  NN::Layer<2, 2> layer;

  EXPECT_DOUBLE_EQ(layer.weights[0], 1.0);
  EXPECT_DOUBLE_EQ(layer.weights[1], 1.0);
  EXPECT_DOUBLE_EQ(layer.weights[2], 1.0);
  EXPECT_DOUBLE_EQ(layer.weights[3], 1.0);
  EXPECT_DOUBLE_EQ(layer.weights[4], 1.0);
  EXPECT_DOUBLE_EQ(layer.weights[5], 1.0);
};

TEST(LayerDefaultConstructor, InitializesLearningRate) {
  NN::Layer<2, 2> layer;

  EXPECT_DOUBLE_EQ(layer.learning_rate, 0.005);
};

TEST(LayerDefaultConstructor, InitializesActivation) {
  NN::Layer<2, 2> layer;

  ASSERT_NE(layer.activation, nullptr);
  EXPECT_EQ(layer.activation->getType(), NN::ActivationType::LINEARTYPE);
};

TEST(LayerDefaultConstructor, InitializesLoss) {
  NN::Layer<2, 2> layer;

  ASSERT_NE(layer.loss, nullptr);
  EXPECT_EQ(layer.loss->getType(), NN::LossType::MSETYPE);
};



// =============================
// ======== Node Control ========
// =============================
TEST(LayerSetNodes, SetsNodes) {
  NN::Layer<3, 1> layer;

  layer.setNodes({1.0, 2.0, 3.0});

  EXPECT_DOUBLE_EQ(layer.nodes[0], 1.0);
  EXPECT_DOUBLE_EQ(layer.nodes[1], 2.0);
  EXPECT_DOUBLE_EQ(layer.nodes[2], 3.0);
};

TEST(LayerSetNodes, IgnoresAdditionalNodes) {
  NN::Layer<2, 1> layer;

  layer.setNodes({1.0, 2.0, 3.0, 4.0});

  EXPECT_DOUBLE_EQ(layer.nodes[0], 1.0);
  EXPECT_DOUBLE_EQ(layer.nodes[1], 2.0);
};

TEST(LayerSetNodes, ReplacesProvidedNodesOnly) {
  NN::Layer<3, 1> layer;
  layer.setNodes({1.0, 2.0, 3.0});

  layer.setNodes({8.0});

  EXPECT_DOUBLE_EQ(layer.nodes[0], 8.0);
  EXPECT_DOUBLE_EQ(layer.nodes[1], 2.0);
  EXPECT_DOUBLE_EQ(layer.nodes[2], 3.0);
};

TEST(LayerGetNodes, ReturnsInternalArray) {
  NN::Layer<2, 1> layer;
  layer.setNodes({4.0, 8.0});

  double* nodes = layer.getNodes();

  ASSERT_NE(nodes, nullptr);
  EXPECT_EQ(nodes, layer.nodes);
  EXPECT_DOUBLE_EQ(nodes[0], 4.0);
  EXPECT_DOUBLE_EQ(nodes[1], 8.0);
};

TEST(LayerOperatorIndex, ReturnsNodeReference) {
  NN::Layer<2, 1> layer;
  layer.setNodes({1.0, 2.0});

  layer[0] = 10.0;

  EXPECT_DOUBLE_EQ(layer.nodes[0], 10.0);
};

TEST(LayerOperatorIndex, AllowsBiasNode) {
  NN::Layer<2, 1> layer;

  layer[2] = 5.0;

  EXPECT_DOUBLE_EQ(layer.nodes[2], 5.0);
};

TEST(LayerOperatorIndex, ThrowsWhenIndexIsOutOfRange) {
  NN::Layer<2, 1> layer;

  EXPECT_THROW(layer[3], std::range_error);
};



// =============================
// ======= Weight Control =======
// =============================
TEST(LayerSetWeights, SetsWeights) {
  NN::Layer<2, 2> layer;

  layer.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  EXPECT_DOUBLE_EQ(layer.weights[0], 1.0);
  EXPECT_DOUBLE_EQ(layer.weights[1], 2.0);
  EXPECT_DOUBLE_EQ(layer.weights[2], 3.0);
  EXPECT_DOUBLE_EQ(layer.weights[3], 4.0);
  EXPECT_DOUBLE_EQ(layer.weights[4], 5.0);
  EXPECT_DOUBLE_EQ(layer.weights[5], 6.0);
};

TEST(LayerSetWeights, IgnoresAdditionalWeights) {
  NN::Layer<1, 1> layer;

  layer.setWeights({2.0, 3.0, 4.0, 5.0});

  EXPECT_DOUBLE_EQ(layer.weights[0], 2.0);
  EXPECT_DOUBLE_EQ(layer.weights[1], 3.0);
};

TEST(LayerSetWeights, ReplacesProvidedWeightsOnly) {
  NN::Layer<2, 2> layer;
  layer.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  layer.setWeights({10.0, 20.0});

  EXPECT_DOUBLE_EQ(layer.weights[0], 10.0);
  EXPECT_DOUBLE_EQ(layer.weights[1], 20.0);
  EXPECT_DOUBLE_EQ(layer.weights[2], 3.0);
  EXPECT_DOUBLE_EQ(layer.weights[3], 4.0);
  EXPECT_DOUBLE_EQ(layer.weights[4], 5.0);
  EXPECT_DOUBLE_EQ(layer.weights[5], 6.0);
};

TEST(LayerSetWeightsPointer, CopiesWeights) {
  NN::Layer<2, 2> layer;

  const double weights[] = {
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  };

  layer.setWeights(weights);

  EXPECT_DOUBLE_EQ(layer.weights[0], 1.0);
  EXPECT_DOUBLE_EQ(layer.weights[1], 2.0);
  EXPECT_DOUBLE_EQ(layer.weights[2], 3.0);
  EXPECT_DOUBLE_EQ(layer.weights[3], 4.0);
  EXPECT_DOUBLE_EQ(layer.weights[4], 5.0);
  EXPECT_DOUBLE_EQ(layer.weights[5], 6.0);
};

TEST(LayerSetWeightsPointer, PerformsCopy) {
  NN::Layer<1, 1> layer;
  double weights[] = {2.0, 4.0};

  layer.setWeights(weights);

  weights[0] = 100.0;
  weights[1] = 200.0;

  EXPECT_DOUBLE_EQ(layer.weights[0], 2.0);
  EXPECT_DOUBLE_EQ(layer.weights[1], 4.0);
};

TEST(LayerGetWeights, ReturnsInternalArray) {
  NN::Layer<2, 2> layer;

  EXPECT_EQ(layer.getWeights(), layer.weights);
};



// =============================
// ======== Activation =========
// =============================
TEST(LayerGetActivatedNode, UsesLinearActivation) {
  NN::Layer<2, 1> layer;
  layer.setNodes({-2.0, 4.0});

  EXPECT_DOUBLE_EQ(layer.getActivatedNode(0), -2.0);
  EXPECT_DOUBLE_EQ(layer.getActivatedNode(1), 4.0);
};

TEST(LayerGetActivatedNode, UsesSigmoidActivation) {
  NN::Layer<2, 1> layer;
  layer.setNodes({0.0, 1.0});

  layer.setActivation(NN::ActivationType::SIGMOIDTYPE);

  EXPECT_DOUBLE_EQ(layer.getActivatedNode(0), 0.5);
  EXPECT_NEAR(layer.getActivatedNode(1), 0.7310585786300049, 1e-12);
};

TEST(LayerSetActivation, SetsSigmoidActivation) {
  NN::Layer<2, 1> layer;

  layer.setActivation(NN::ActivationType::SIGMOIDTYPE);

  ASSERT_NE(layer.activation, nullptr);
  EXPECT_EQ(layer.activation->getType(), NN::ActivationType::SIGMOIDTYPE);
};

TEST(LayerSetActivation, ReplacesPreviousActivation) {
  NN::Layer<2, 1> layer;

  layer.setActivation(NN::ActivationType::SIGMOIDTYPE);
  layer.setActivation(NN::ActivationType::LINEARTYPE);

  ASSERT_NE(layer.activation, nullptr);
  EXPECT_EQ(layer.activation->getType(), NN::ActivationType::LINEARTYPE);
};

TEST(LayerGetActivation, ReturnsStoredActivation) {
  NN::Layer<2, 1> layer;

  EXPECT_EQ(layer.getActivation(), layer.activation);
};



// =============================
// =========== Loss ============
// =============================
TEST(LayerSetLoss, SetsMSELoss) {
  NN::Layer<2, 1> layer;

  layer.setLoss(NN::LossType::MSETYPE);

  ASSERT_NE(layer.loss, nullptr);
  EXPECT_EQ(layer.loss->getType(), NN::LossType::MSETYPE);
};

TEST(LayerGetLoss, ReturnsStoredLoss) {
  NN::Layer<2, 1> layer;

  EXPECT_EQ(layer.getLoss(), layer.loss);
};



// =============================
// ======= Learning Rate =======
// =============================
TEST(LayerGetLearningRate, ReturnsDefaultLearningRate) {
  NN::Layer<2, 1> layer;

  EXPECT_DOUBLE_EQ(layer.getLearningRate(), 0.005);
};

TEST(LayerSetLearningRate, SetsLearningRate) {
  NN::Layer<2, 1> layer;

  layer.setLearningRate(0.001);

  EXPECT_DOUBLE_EQ(layer.getLearningRate(), 0.001);
};

TEST(LayerSetLearningRate, ReplacesLearningRate) {
  NN::Layer<2, 1> layer;

  layer.setLearningRate(0.1);
  layer.setLearningRate(0.0001);

  EXPECT_DOUBLE_EQ(layer.getLearningRate(), 0.0001);
};



// =============================
// ========== Forward ==========
// =============================
TEST(LayerForward, CalculatesSingleDestinationNode) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> destination;

  source.setNodes({2.0, 3.0});
  source.setWeights({4.0, 5.0, 6.0});

  source.forward(destination);

  EXPECT_DOUBLE_EQ(destination[0], 29.0);
};

TEST(LayerForward, CalculatesMultipleDestinationNodes) {
  NN::Layer<2, 2> source;
  NN::Layer<2, 0> destination;

  source.setNodes({2.0, 3.0});
  source.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  source.forward(destination);

  EXPECT_DOUBLE_EQ(destination[0], 11.0);
  EXPECT_DOUBLE_EQ(destination[1], 29.0);
};

TEST(LayerForward, SetsSourceBiasNode) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> destination;

  source.setNodes({2.0, 3.0});

  source.forward(destination);

  EXPECT_DOUBLE_EQ(source.nodes[2], 1.0);
};

TEST(LayerForward, SetsDestinationBiasNode) {
  NN::Layer<2, 2> source;
  NN::Layer<2, 0> destination;

  source.setNodes({2.0, 3.0});

  source.forward(destination);

  EXPECT_DOUBLE_EQ(destination.nodes[2], 1.0);
};

TEST(LayerForward, IncludesBiasWeight) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> destination;

  source.setNodes({0.0, 0.0});
  source.setWeights({0.0, 0.0, 7.0});

  source.forward(destination);

  EXPECT_DOUBLE_EQ(destination[0], 7.0);
};

TEST(LayerForward, HandlesNegativeWeights) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> destination;

  source.setNodes({2.0, 3.0});
  source.setWeights({-2.0, 4.0, -1.0});

  source.forward(destination);

  EXPECT_DOUBLE_EQ(destination[0], 7.0);
};

TEST(LayerForward, AppliesActivationBeforeWeights) {
  NN::Layer<1, 1> source;
  NN::Layer<1, 0> destination;

  source.setNodes({0.0});
  source.setWeights({2.0, 0.0});
  source.setActivation(NN::ActivationType::SIGMOIDTYPE);

  source.forward(destination);

  EXPECT_DOUBLE_EQ(destination[0], 1.0);
};

TEST(LayerForward, ProducesFiniteValues) {
  NN::Layer<3, 3> source;
  NN::Layer<3, 0> destination;

  source.setNodes({0.1, 0.2, 0.3});
  source.setWeights({
    0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.1, 1.2
  });

  source.forward(destination);

  EXPECT_TRUE(std::isfinite(destination[0]));
  EXPECT_TRUE(std::isfinite(destination[1]));
  EXPECT_TRUE(std::isfinite(destination[2]));
};



// =============================
// ===== Backprop Initial ======
// =============================
TEST(LayerBackpropInitial, CalculatesOutputSigma) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> output;

  source.setNodes({2.0, 3.0});
  source.setWeights({1.0, 1.0, 0.0});

  source.forward(output);
  output.backprop_initial(source, {1.0});

  EXPECT_DOUBLE_EQ(output.sigma[0], 4.0);
};

TEST(LayerBackpropInitial, CalculatesZeroSigmaForCorrectPrediction) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> output;

  source.setNodes({2.0, 3.0});
  source.setWeights({1.0, 1.0, 0.0});

  source.forward(output);
  output.backprop_initial(source, {5.0});

  EXPECT_DOUBLE_EQ(output.sigma[0], 0.0);
};

TEST(LayerBackpropInitial, CorrectlyUpdatesPreviousWeights) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> output;

  source.setNodes({2.0, 3.0});
  source.setWeights({1.0, 1.0, 0.0});

  output.setLearningRate(0.1);

  source.forward(output);
  output.backprop_initial(source, {1.0});

  EXPECT_NEAR(source.weights[0], 0.2, 1e-12);
  EXPECT_NEAR(source.weights[1], -0.2, 1e-12);
  EXPECT_NEAR(source.weights[2], -0.4, 1e-12);
};

TEST(LayerBackpropInitial, DoesNotChangeWeightsWhenLossIsZero) {
  NN::Layer<2, 1> source;
  NN::Layer<1, 0> output;

  source.setNodes({2.0, 3.0});
  source.setWeights({1.0, 1.0, 0.0});

  source.forward(output);
  output.backprop_initial(source, {5.0});

  EXPECT_DOUBLE_EQ(source.weights[0], 1.0);
  EXPECT_DOUBLE_EQ(source.weights[1], 1.0);
  EXPECT_DOUBLE_EQ(source.weights[2], 0.0);
};



// =============================
// ========= Backprop ==========
// =============================
TEST(LayerBackprop, CalculatesSigma) {
  NN::Layer<2, 2> source;
  NN::Layer<2, 0> next;

  source.setNodes({2.0, 3.0});
  source.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  next.sigma[0] = 0.5;
  next.sigma[1] = -1.0;

  source.backprop(next);

  EXPECT_DOUBLE_EQ(source.sigma[0], -3.5);
  EXPECT_DOUBLE_EQ(source.sigma[1], -4.0);
};

TEST(LayerBackprop, UpdatesWeights) {
  NN::Layer<2, 2> source;
  NN::Layer<2, 0> next;

  source.setLearningRate(0.1);
  source.setNodes({2.0, 3.0});
  source.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  next.sigma[0] = 0.5;
  next.sigma[1] = -1.0;

  source.backprop(next);

  EXPECT_DOUBLE_EQ(source.weights[0], 0.9);
  EXPECT_DOUBLE_EQ(source.weights[1], 1.85);
  EXPECT_DOUBLE_EQ(source.weights[2], 2.95);

  EXPECT_DOUBLE_EQ(source.weights[3], 4.2);
  EXPECT_DOUBLE_EQ(source.weights[4], 5.3);
  EXPECT_DOUBLE_EQ(source.weights[5], 6.1);
};

TEST(LayerBackprop, DoesNotChangeWeightsForZeroSigma) {
  NN::Layer<2, 2> source;
  NN::Layer<2, 0> next;

  source.setLearningRate(0.1);
  source.setNodes({2.0, 3.0});
  source.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  next.sigma[0] = 0.0;
  next.sigma[1] = 0.0;

  source.backprop(next);

  EXPECT_DOUBLE_EQ(source.weights[0], 1.0);
  EXPECT_DOUBLE_EQ(source.weights[1], 2.0);
  EXPECT_DOUBLE_EQ(source.weights[2], 3.0);
  EXPECT_DOUBLE_EQ(source.weights[3], 4.0);
  EXPECT_DOUBLE_EQ(source.weights[4], 5.0);
  EXPECT_DOUBLE_EQ(source.weights[5], 6.0);
};

TEST(LayerBackprop, KeepsValuesFinite) {
  NN::Layer<2, 2> source;
  NN::Layer<2, 0> next;

  source.setLearningRate(0.001);
  source.setNodes({0.25, 0.5});

  next.sigma[0] = 0.2;
  next.sigma[1] = -0.1;

  source.backprop(next);

  EXPECT_TRUE(std::isfinite(source.sigma[0]));
  EXPECT_TRUE(std::isfinite(source.sigma[1]));

  for(double weight : source.weights)
    EXPECT_TRUE(std::isfinite(weight));
};



// =============================
// =========== Sigma ===========
// =============================
TEST(LayerGetSigma, ReturnsInternalArray) {
  NN::Layer<2, 1> layer;

  layer.sigma[0] = 2.0;
  layer.sigma[1] = 4.0;

  const double* sigma = layer.getSigma();

  ASSERT_NE(sigma, nullptr);
  EXPECT_EQ(sigma, layer.sigma);
  EXPECT_DOUBLE_EQ(sigma[0], 2.0);
  EXPECT_DOUBLE_EQ(sigma[1], 4.0);
};



// =============================
// =========== Print ===========
// =============================
TEST(LayerPrint, ContainsDimensions) {
  NN::Layer<2, 1> layer;
  layer.setNodes({1.0, 2.0});
  layer[2] = 1.0;

  const std::string data = layer.print();

  EXPECT_NE(data.find("2, 1"), std::string::npos);
};

TEST(LayerPrint, ContainsNodes) {
  NN::Layer<2, 1> layer;
  layer.setNodes({2.0, 4.0});
  layer[2] = 1.0;

  const std::string data = layer.print();

  EXPECT_NE(data.find("2.000000"), std::string::npos);
  EXPECT_NE(data.find("4.000000"), std::string::npos);
};

TEST(LayerPrint, ContainsWeights) {
  NN::Layer<2, 1> layer;
  layer.setNodes({1.0, 2.0});
  layer[2] = 1.0;
  layer.setWeights({3.0, 4.0, 5.0});

  const std::string data = layer.print();

  EXPECT_NE(data.find("3.000000"), std::string::npos);
  EXPECT_NE(data.find("4.000000"), std::string::npos);
  EXPECT_NE(data.find("5.000000"), std::string::npos);
};



// =============================
// ======= Serialization =======
// =============================
TEST(LayerSerialize, SerializesLayer) {
  NN::Layer<2, 1> layer;

  layer.setNodes({1.0, 2.0});
  layer[2] = 1.0;
  layer.setWeights({3.0, 4.0, 5.0});

  const std::string data = layer.serialize();

  EXPECT_EQ(
    data,
    "2, 1; 0; 0; 1.000000, 2.000000, 1.000000; 3.000000, 4.000000, 5.000000; "
  );
};

TEST(LayerSerialize, ReflectsChangedActivation) {
  NN::Layer<1, 1> layer;

  layer.setNodes({0.0});
  layer[1] = 1.0;
  layer.setWeights({1.0, 2.0});
  layer.setActivation(NN::ActivationType::SIGMOIDTYPE);

  const std::string data = layer.serialize();

  EXPECT_NE(data.find("1; 0.000000"), std::string::npos);
};

TEST(LayerSerialize, ReflectsWeights) {
  NN::Layer<1, 1> layer;

  layer.setNodes({1.0});
  layer[1] = 1.0;
  layer.setWeights({12.0, -4.0});

  const std::string data = layer.serialize();

  EXPECT_NE(data.find("12.000000"), std::string::npos);
  EXPECT_NE(data.find("-4.000000"), std::string::npos);
};