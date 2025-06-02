#include "Node.hpp"
#include "UnitTests.h"
#include "Exampe.h"
#include <cstdio>

int main(){
  UnitTests::runAllTests();
  Examples::NodeExample();

  NeuralNetwork::Node<0> node1({0.0f}, 1.0f);
  printf("%s\n", node1.serialize().c_str());
  NeuralNetwork::Node<0> node2(node1.serialize().c_str());
  printf("%s\n", node2.serialize().c_str());

  return 0;
}