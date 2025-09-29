#pragma once
#include "dsl.h"

namespace Examples{
void NeuralNetwork() {
  fmt::print(fg(fmt::color::aquamarine) , "=== NeuralNetwork Example ===\n");

  // create Layer
  NN::Layer<1, 1> layer;
  fmt::print(fg(fmt::color::violet), "{}\n\n", layer.print());
  layer.setNodes({6});
  fmt::print(fg(fmt::color::violet), "{}\n\n", layer.print());
  layer.setWeights({2, 4});
  fmt::print(fg(fmt::color::violet), "{}\n\n", layer.print());

};
};