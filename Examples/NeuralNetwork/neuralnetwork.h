#pragma once
#include "dsl.h"

namespace Examples{
void NeuralNetwork() {
  fmt::print(fg(fmt::color::aquamarine) , "=== NeuralNetwork Example ===\n");

  // create Layer
  fmt::print(fg(fmt::color::crimson), "create Layer\n");
  NN::Layer<2, 1> layer;
  fmt::print(fg(fmt::color::violet), "{}\n", layer.print());
  
  // set Nodes
  fmt::print(fg(fmt::color::crimson), "set Nodes\n");
  layer.setNodes({6});
  fmt::print(fg(fmt::color::violet), "{}\n", layer.print());
  
  // set Wights
  fmt::print(fg(fmt::color::crimson), "set Weights\n");
  layer.setWeights({2, 4});
  fmt::print(fg(fmt::color::violet), "{}\n", layer.print());

  // set Loss
  fmt::print(fg(fmt::color::crimson), "set Loss\n");
  layer.setLoss(NN::LossType::MSETYPE);
  fmt::print(fg(fmt::color::violet), "{}\n", layer.print());

  // set Activation
  fmt::print(fg(fmt::color::crimson), "set Activation\n");
  layer.setActivation(NN::ActivationType::SIGMOIDTYPE);
  fmt::print(fg(fmt::color::violet), "{}\n", layer.print());
  layer.setActivation(NN::ActivationType::NOTYPE);
  fmt::print(fg(fmt::color::violet), "{}\n", layer.print());
  
  // forward
  fmt::print(fg(fmt::color::crimson), "forward\n");
  layer.setNodes({5, 2});
  layer.setWeights({1, 2, 6});
  NN::Layer<1, 0> layer2;
  layer.forward(layer2); // 5 * 1 + 2 * 2 + 6 = 15
  fmt::print(fg(fmt::color::violet), "{}\n", layer2.print());

};
};