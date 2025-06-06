#include "Node.hpp"
#include <iostream>

namespace Examples{
  void NodeExample(){
    std::cout << "------------------- [Node Example] -------------------\n";
    // Define weights (for 3 inputs + bias)
    std::array<float, 4> weights = {0.2f, -0.5f, 1.0f, 0.1f};
    // Initial value (can be 0)
    float initialValue = 0.0f;

    // Create Node with weights and value
    NeuralNetwork::Node<3> node(weights, initialValue);

    // Input values from previous layer
    std::array<float, 3> inputs = {1.0f, 2.0f, -1.0f};

    // Calculate node's value using inputs and weights
    node.calculate(inputs);

    std::cout << "Calculated Node value: " << node.getValue() << std::endl;

    // Serialize the node state to string
    std::string serialized = node.serialize();
    std::cout << "Serialized node: " << serialized << std::endl;

    // Create a new node and deserialize the string into it
    NeuralNetwork::Node<3> node2;
    node2.deserialize(serialized);

    std::cout << "Deserialized Node value: " << node2.getValue() << std::endl;

    // Confirm weights are equal
    const auto& w1 = node.getWeights();
    const auto& w2 = node2.getWeights();

    bool weightsEqual = true;
    for (size_t i = 0; i < w1.size(); ++i) {
        if (w1[i] != w2[i]) {
            weightsEqual = false;
            break;
        }
    }
    std::cout << "Weights equal after deserialize? " << (weightsEqual ? "Yes" : "No") << std::endl;
  }

} // namespace NodeExample
