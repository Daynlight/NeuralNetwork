#pragma once
#include "dsl.h"

namespace Examples{
void NeuralNetwork() {
  fmt::print(fg(fmt::color::aquamarine) , "=== NeuralNetwork Example ===\n");

  // create Layer
  NN::Layer<2, 1> layer;
  fmt::print("{}\n", layer.serialize());

  fmt::print("\n\n");
};
};