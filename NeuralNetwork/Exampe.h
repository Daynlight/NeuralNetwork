#include "Node.hpp"
#include "Layer.hpp"
#include <iostream>
#include <vector>

namespace Examples {

void NodeExample() {
  std::cout << "------------------- [Node Example] -------------------\n";

  // Define weights for 3 inputs + bias (4 weights total)
  std::vector<float> weights = {0.2f, -0.5f, 1.0f, 0.1f};
  float initialValue = 0.0f;

  NeuralNetwork::Node node(weights, initialValue);

  std::vector<float> inputs = {1.0f, 2.0f, -1.0f};
  node.calculate(inputs);

  std::cout << "Calculated Node value: " << node.getValue() << "\n";

  std::string serialized = node.serialize();
  std::cout << "Serialized node: " << serialized << "\n";

  NeuralNetwork::Node node2(weights.size() - 1); // construct empty node with input size
  node2.deserialize(serialized);

  std::cout << "Deserialized Node value: " << node2.getValue() << "\n";

  const auto& w1 = node.getWeights();
  const auto& w2 = node2.getWeights();

  bool weightsEqual = (w1.size() == w2.size());
  for (size_t i = 0; weightsEqual && i < w1.size(); ++i) {
    if (w1[i] != w2[i]) weightsEqual = false;
  }
  std::cout << "Weights equal after deserialize? " << (weightsEqual ? "Yes" : "No") << "\n";
}

void LayerExample() {
  std::cout << "------------------- [Layer Example] -------------------\n";

  // --- Input Layer ---
  // Create 2 input nodes (no inputs themselves, just values)
  NeuralNetwork::Layer inputLayer(0, 2);
  std::vector<NeuralNetwork::Node> inputNodes(2, NeuralNetwork::Node(0)); // no weights, just inputs
  inputNodes[0].setValue(1.5f);
  inputNodes[1].setValue(-0.5f);
  inputLayer.setNodes(inputNodes);

  // --- Hidden Layer ---
  // 2 nodes, each with 2 inputs + bias (3 weights)
  NeuralNetwork::Layer hiddenLayer(&inputLayer, 2, 2);
  std::vector<NeuralNetwork::Node> hiddenNodes;
  hiddenNodes.emplace_back(std::vector<float>{0.5f, -1.0f, 0.1f});
  hiddenNodes.emplace_back(std::vector<float>{1.5f, 0.5f, -0.3f});
  hiddenLayer.setNodes(hiddenNodes);

  hiddenLayer.calculate();

  auto hiddenValues = hiddenLayer.getNodesValues();
  std::cout << "Calculated hidden layer node values:\n";
  for (size_t i = 0; i < hiddenValues.size(); ++i) {
    std::cout << "  Node " << i << ": " << hiddenValues[i] << "\n";
  }

  // --- Output Layer ---
  // 1 node, with 2 inputs + bias
  NeuralNetwork::Layer outputLayer(&hiddenLayer, 2, 1);
  std::vector<NeuralNetwork::Node> outputNodes;
  outputNodes.emplace_back(std::vector<float>{-1.0f, 2.0f, 0.5f});
  outputLayer.setNodes(outputNodes);

  outputLayer.calculate();

  auto outputValues = outputLayer.getNodesValues();
  std::cout << "Calculated output layer node value:\n";
  std::cout << "  Node 0: " << outputValues[0] << "\n";

  std::string serialized = outputLayer.serialize();
  std::cout << "Serialized output layer: " << serialized << "\n";

  NeuralNetwork::Layer outputLayer2(&hiddenLayer, 2, 1);
  outputLayer2.deserialize(serialized);
  outputLayer2.calculate();

  auto outputValues2 = outputLayer2.getNodesValues();
  std::cout << "Deserialized and recalculated output layer node value:\n";
  std::cout << "  Node 0: " << outputValues2[0] << "\n";
}

}  // namespace Examples
