#include "Node.hpp"
#include <cassert>
#include <array>
#include <string>
#include <stdexcept>
#include <iostream>

/*
Unit Tests for NeuralNetwork::Node

- testDefaultConstructor: Checks zero initialization.
- testConstructorWithWeightsAndValue: Validates constructor initialization.
- testCalculate: Validates the weighted sum + bias calculation.
- testSerializeDeserialize: Ensures serialization round-trip correctness.
- testGetSetData: Tests batch data get/set.
- testDeserializeInvalidThrows: Ensures exceptions on invalid input.

Tests include both normal cases and error handling for robustness.
*/

namespace UnitTests
{
// ################### [NeuralNetwork::Node] ################### //
void testDefaultConstructor() {
    NeuralNetwork::Node<3> node;
    assert(node.getValue() == 0.0f);
    for (auto w : node.getWeights())
        assert(w == 0.0f);
};

void testConstructorWithWeightsAndValue() {
    std::array<float, 4> weights = {1.0f, 2.0f, 3.0f, 4.0f};
    float value = 5.0f;
    NeuralNetwork::Node<3> node(weights, value);

    assert(node.getValue() == value);
    assert(node.getWeights() == weights);
};

void testCalculate() {
    std::array<float, 3> prevValues = {1.0f, 2.0f, 3.0f};
    std::array<float, 4> weights = {0.5f, 1.0f, -1.5f, 2.0f}; // last is bias
    NeuralNetwork::Node<3> node(weights, 0.0f);

    node.calculate(prevValues);

    // Expected: 1*0.5 + 2*1.0 + 3*(-1.5) + 2.0 = 0.5 + 2 - 4.5 + 2 = 0
    assert(node.getValue() == 0.0f);
};

void testSerializeDeserialize() {
    std::array<float, 4> weights = {1.1f, 2.2f, 3.3f, 4.4f};
    float value = 5.5f;

    NeuralNetwork::Node<3> node(weights, value);
    std::string serialized = node.serialize();

    NeuralNetwork::Node<3> newNode;
    newNode.deserialize(serialized);

    assert(newNode.getValue() == value);
    assert(newNode.getWeights() == weights);
};

void testGetSetData() {
    std::array<float, 5> data = {7.7f, 1.0f, 2.0f, 3.0f, 4.0f};
    NeuralNetwork::Node<3> node;
    node.setData(data);

    assert(node.getValue() == 7.7f);

    auto weights = node.getWeights();
    for (size_t i = 0; i < 4; i++) {
        assert(weights[i] == data[i + 1]);
    }

    auto retrievedData = node.getData();
    assert(retrievedData == data);
};

void testDeserializeInvalidThrows() {
    NeuralNetwork::Node<3> node;

    bool caught = false;
    try {
        node.deserialize("1.0;2.0;3.0;"); // too few tokens
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        node.deserialize("a;2.0;3.0;4.0;5.0;"); // invalid float
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
};

// Helper to run all tests and print summary
void runAllTests() {
    testDefaultConstructor();
    testConstructorWithWeightsAndValue();
    testCalculate();
    testSerializeDeserialize();
    testGetSetData();
    testDeserializeInvalidThrows();

    std::cout << "All Node tests passed!" << std::endl;
};
};