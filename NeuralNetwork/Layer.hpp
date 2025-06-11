#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "Node.hpp"

namespace NeuralNetwork {

class Layer {
private:
  std::size_t W = 0;  // input size per node (weights without bias)
  std::size_t S = 0;  // number of nodes

  std::vector<Node> nodes;
  Layer* last = nullptr;
  Layer* next = nullptr;

public:
  // [Constructors]
  Layer(std::size_t inputSize, std::size_t nodeCount)
    : W(inputSize), S(nodeCount), nodes(S, Node(W)) {}

  Layer(Layer* lastLayer, std::size_t inputSize, std::size_t nodeCount)
    : W(inputSize), S(nodeCount), nodes(S, Node(W)), last(lastLayer) {
    if (last) last->next = this;
  }

  Layer(Layer* lastLayer, const std::vector<Node>& nodesVec)
    : W(0), S(nodesVec.size()), nodes(nodesVec), last(lastLayer) {
    if (last) last->next = this;
    if (S > 0) W = nodes[0].getWeights().size() - 1;
  }

  // [Getters/Setters]
  std::size_t getInputSize() const noexcept { return W; }
  std::size_t getNodeCount() const noexcept { return S; }

  const std::vector<Node>& getNodes() const noexcept { return nodes; }
  void setNodes(const std::vector<Node>& newNodes) {
    nodes = newNodes;
    S = nodes.size();
    if (S > 0)
      W = nodes[0].getWeights().size() - 1;
  }

  std::vector<float> getNodesValues() const noexcept {
    std::vector<float> values(S);
    for (std::size_t i = 0; i < S; ++i)
      values[i] = nodes[i].getValue();
    return values;
  }

  void setNodesValues(const std::vector<float>& values) {
    if (values.size() != S)
      throw std::runtime_error("Values size mismatch in setNodesValues");
    for (std::size_t i = 0; i < S; ++i)
      nodes[i].setValue(values[i]);
  }

  std::vector<float> getNodesWeights() const noexcept {
    std::vector<float> weights(S * (W + 1));
    for (std::size_t i = 0; i < S; ++i) {
      const auto& w = nodes[i].getWeights();
      std::copy(w.begin(), w.end(), weights.begin() + i * (W + 1));
    }
    return weights;
  }

  void setNodesWeights(const std::vector<float>& weights) {
    if (weights.size() != S * (W + 1))
      throw std::runtime_error("Weights size mismatch in setNodesWeights");
    for (std::size_t i = 0; i < S; ++i) {
      std::vector<float> w(weights.begin() + i * (W + 1), weights.begin() + (i + 1) * (W + 1));
      nodes[i].setWeights(w);
    }
  }

  const Layer* getLastLayer() const noexcept { return last; }
  const Layer* getNextLayer() const noexcept { return next; }

  // [Calculate]
  void calculate() {
    if (!last) return;
    std::vector<float> inputs = last->getNodesValues();
    for (auto& node : nodes) node.calculate(inputs);
  }

  // [Get Batch/Load Batch]
  std::vector<float> getData() const noexcept {
    std::vector<float> data(S * (W + 2));
    for (std::size_t i = 0; i < S; ++i) {
      auto nodeData = nodes[i].getData();
      std::copy(nodeData.begin(), nodeData.end(), data.begin() + i * (W + 2));
    }
    return data;
  }

  void setData(const std::vector<float>& data) {
    if (data.size() != S * (W + 2))
      throw std::runtime_error("Data size mismatch in setData");
    for (std::size_t i = 0; i < S; ++i) {
      std::vector<float> nodeData(data.begin() + i * (W + 2), data.begin() + (i + 1) * (W + 2));
      nodes[i].setData(nodeData);
    }
  }

  // [Serialization/Deserialization]
  std::string serialize() const {
    std::ostringstream str;
    for (const auto& node : nodes)
      str << node.serialize();
    return str.str();
  }

  void deserialize(const std::string& str) {
    std::vector<std::string> nodeStrings;
    std::size_t semicolonCount = 0;
    std::size_t start = 0;

    // Determine size of data per node: W + 2 floats, each ended by ';'
    const std::size_t tokensPerNode = W + 2;

    for (std::size_t i = 0; i < str.size(); ++i) {
      if (str[i] == ';') {
        ++semicolonCount;
        if (semicolonCount == tokensPerNode) {
          nodeStrings.push_back(str.substr(start, i - start + 1));
          start = i + 1;
          semicolonCount = 0;
        }
      }
    }

    if (nodeStrings.size() != S)
      throw std::runtime_error("Incorrect number of nodes in deserialize");

    for (std::size_t i = 0; i < S; ++i)
      nodes[i].deserialize(nodeStrings[i]);
  }
};

} // namespace NeuralNetwork