#include "Node.hpp"
#include "Layer.hpp"
#include <cassert>
#include <array>
#include <string>
#include <stdexcept>
#include <iostream>

namespace UnitTests {
    // ################### [NeuralNetwork::Node] ################### //
    namespace NodeTests {
    // Test default constructor - Note: your class has no default constructor, so let's test the one with input size 0
    void testConstructorWithInputSize() {
        NeuralNetwork::Node node(0);
        assert(node.getValue() == 0.0f);
        assert(node.getWeights().size() == 1); // weights size = inputSize + 1, so 0+1 = 1 (bias)
        for (auto w : node.getWeights())
            assert(w == 0.0f);
    }

    void testConstructorWithWeightsAndValue() {
        std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};
        float value = 5.0f;
        NeuralNetwork::Node node(weights, value);

        assert(node.getValue() == value);
        assert(node.getWeights() == weights);
    }

    void testConstructorFromData() {
        std::vector<float> data = {9.9f, 1.1f, 2.2f, 3.3f, 4.4f};
        NeuralNetwork::Node node(data, true);

        assert(node.getValue() == 9.9f);
        auto weights = node.getWeights();
        for (size_t i = 0; i < weights.size(); i++)
            assert(weights[i] == data[i + 1]);
    }

    void testConstructorFromStr() {
        std::string serialized = "6.6;0.1;0.2;0.3;0.4;";
        NeuralNetwork::Node node(serialized);

        assert(node.getValue() == 6.6f);
        std::vector<float> expectedWeights = {0.1f, 0.2f, 0.3f, 0.4f};
        assert(node.getWeights() == expectedWeights);
    }

    void testGettersAndSetters() {
        NeuralNetwork::Node node(4); // input size 4
        node.setValue(8.8f);
        std::vector<float> newWeights = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
        node.setWeights(newWeights);

        assert(node.getValue() == 8.8f);
        assert(node.getWeights() == newWeights);
    }

    void testCalculate() {
        std::vector<float> prevValues = {1.0f, 2.0f, 3.0f};
        std::vector<float> weights = {0.5f, 1.0f, -1.5f, 2.0f}; // last is bias
        NeuralNetwork::Node node(weights, 0.0f);

        node.calculate(prevValues);

        // Expected: 1*0.5 + 2*1.0 + 3*(-1.5) + 2.0 = 0.5 + 2 - 4.5 + 2 = 0
        assert(node.getValue() == 0.0f);
    }

    void testSerializeDeserialize() {
        std::vector<float> weights = {1.1f, 2.2f, 3.3f, 4.4f};
        float value = 5.5f;

        NeuralNetwork::Node node(weights, value);
        std::string serialized = node.serialize();

        NeuralNetwork::Node newNode(4); // input size 4
        newNode.deserialize(serialized);

        assert(newNode.getValue() == value);
        assert(newNode.getWeights() == weights);
    }

    void testGetSetData() {
        // input size = 4, so data size must be 6 = 1(value) + 4(weights) + 1(bias)
        std::vector<float> data = {7.7f, 1.0f, 2.0f, 3.0f, 4.0f, 0.5f}; // added 0.5f bias

        NeuralNetwork::Node node(4); // input size 4
        node.setData(data);

        assert(node.getValue() == 7.7f);

        auto weights = node.getWeights();
        for (size_t i = 0; i < weights.size(); i++)
            assert(weights[i] == data[i + 1]);

        auto retrievedData = node.getData();
        assert(retrievedData == data);
    }

    void testDeserializeInvalidThrows() {
        NeuralNetwork::Node node(1); // input size 1

        bool caught = false;
        try {
            node.deserialize("1.0;"); // too few tokens
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
    }

    } // namespace NodeTests

    namespace LayerTests {
    // ################### [NeuralNetwork::Layer] ################### //
    void testCalculate() {
        // Setup input layer with 2 nodes (values set manually)
        std::vector<NeuralNetwork::Node> inputNodes{
            NeuralNetwork::Node(0), NeuralNetwork::Node(0)
        };
        inputNodes[0].setValue(1.0f);
        inputNodes[1].setValue(2.0f);

        NeuralNetwork::Layer inputLayer(nullptr, 0, 2);
        inputLayer.setNodes(inputNodes);

        // Setup hidden layer with weights to simulate identity mapping
        std::vector<NeuralNetwork::Node> hiddenNodes;
        hiddenNodes.emplace_back(NeuralNetwork::Node(std::vector<float>{1.0f, 0.0f, 0.0f}));
        hiddenNodes.emplace_back(NeuralNetwork::Node(std::vector<float>{0.0f, 1.0f, 0.0f}));

        NeuralNetwork::Layer hiddenLayer(&inputLayer, 2, 2);
        hiddenLayer.setNodes(hiddenNodes);

        hiddenLayer.calculate();

        auto result = hiddenLayer.getNodesValues();
        assert(result.size() == 2);
        assert(result[0] == 1.0f);
        assert(result[1] == 2.0f);
    }

    void testSerializeDeserialize() {
        // Setup Layer with nodes (weights + values)
        std::vector<NeuralNetwork::Node> nodes{
            NeuralNetwork::Node(std::vector<float>{1.0f, 2.0f, 3.0f}),
            NeuralNetwork::Node(std::vector<float>{5.0f, 6.0f, 7.0f})
        };
        nodes[0].setValue(4.0f);
        nodes[1].setValue(8.0f);

        NeuralNetwork::Layer layer(nullptr, 2, 2);
        layer.setNodes(nodes);

        std::string serialized = layer.serialize();

        NeuralNetwork::Layer newLayer(nullptr, 2, 2);
        newLayer.deserialize(serialized);

        auto originalValues = layer.getNodesValues();
        auto newValues = newLayer.getNodesValues();

        for (size_t i = 0; i < 2; ++i) {
            assert(originalValues[i] == newValues[i]);
            auto originalWeights = layer.getNodes()[i].getWeights();
            auto newWeights = newLayer.getNodes()[i].getWeights();
            assert(originalWeights == newWeights);
        }
    }

    }  // namespace LayerTests

    // Helper to run all tests and print summary
    void runAllTests() {
        std::cout << "------------------- [Unit Tests] -------------------\n";
        std::cout << "------------------- [Node Tests] -------------------\n";
        NodeTests::testConstructorWithWeightsAndValue();
        NodeTests::testConstructorFromData();
        NodeTests::testConstructorFromStr();
        NodeTests::testGettersAndSetters();
        NodeTests::testCalculate();
        NodeTests::testSerializeDeserialize();
        NodeTests::testGetSetData();
        NodeTests::testDeserializeInvalidThrows();
        std::cout << "Node Tests Passed\n";
        std::cout << "------------------- [Layer Tests] -------------------\n";
        LayerTests::testCalculate();
        LayerTests::testSerializeDeserialize();
        std::cout << "Layer Tests Passed\n";

        std::cout << "All tests passed!" << std::endl;
    };
};
