#include "Node.h"
#include <cstdio>

int main(){
  NeuralNetwork::Node<0> node1({} ,0.0f, 1.0f);
  printf("%s\n", node1.getDataString().c_str());
  NeuralNetwork::Node<0> node2(node1.getDataString());
  printf("%s\n", node2.getDataString().c_str());

  return 0;
}