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
- testConstructorFromData: Validates constructor initialization with data array.
- testConstructorFromStr: Validates constructor initialization with string.
- testGettersAndSetters: Tests getters and setters methods.
- testCalculate: Validates the weighted sum + bias calculation.
- testSerializeDeserialize: Ensures serialization round-trip correctness.
- testGetSetData: Tests batch data get/set.
- testDeserializeInvalidThrows: Ensures exceptions on invalid input.

Tests include both normal cases and error handling for robustness.
*/

namespace UnitTests{
// ################### [NeuralNetwork::Node] ################### //
    namespace NodeTests{

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

    void testConstructorFromData() {
        std::array<float, 5> data = {9.9f, 1.1f, 2.2f, 3.3f, 4.4f};
        NeuralNetwork::Node<3> node(data);

        assert(node.getValue() == 9.9f);
        auto weights = node.getWeights();
        for (size_t i = 0; i < 4; i++)
            assert(weights[i] == data[i + 1]);
    };

    void testConstructorFromStr() {
        std::string serialized = "6.6;0.1;0.2;0.3;0.4;";
        NeuralNetwork::Node<3> node(serialized);

        assert(node.getValue() == 6.6f);
        std::array<float, 4> expectedWeights = {0.1f, 0.2f, 0.3f, 0.4f};
        assert(node.getWeights() == expectedWeights);
    };

    void testGettersAndSetters() {
        NeuralNetwork::Node<3> node;
        node.setValue(8.8f);
        std::array<float, 4> newWeights = {0.5f, 1.5f, 2.5f, 3.5f};
        node.setWeights(newWeights);

        assert(node.getValue() == 8.8f);
        assert(node.getWeights() == newWeights);
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
        for (size_t i = 0; i < 4; i++)
            assert(weights[i] == data[i + 1]);

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
    }; // namespace NodeTests

// Helper to run all tests and print summary
void runAllTests() {
    std::cout << "------------------- [Unit Tests] -------------------\n";
    std::cout << "------------------- [Node Tests] -------------------\n";
    NodeTests::testDefaultConstructor();
    NodeTests::testConstructorWithWeightsAndValue();
    NodeTests::testConstructorFromData();
    NodeTests::testConstructorFromStr();
    NodeTests::testGettersAndSetters();
    NodeTests::testCalculate();
    NodeTests::testSerializeDeserialize();
    NodeTests::testGetSetData();
    NodeTests::testDeserializeInvalidThrows();

    std::cout << "All Node tests passed!" << std::endl;
};
}; // namespace UnitTests