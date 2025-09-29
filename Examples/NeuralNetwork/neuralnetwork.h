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
  NN::Layer<4, 3> a;
  NN::Layer<3, 0> b;
  a.setNodes({5, 2, 1, 3});
  a.setWeights({1, 2 ,1, -1, 5,   2, 1, 3, 1, 0,   1, -4, 2, 5, 6});
  // a.setActivation(NN::ActivationType::SIGMOIDTYPE);
  a.forward(b);
  // 5 * 1 + 2 * 2 + 1 * 1 + 3 * (-1) + 5 = 12
  // 5 * 2 + 2 * 1 + 1 * 3 + 3 * 1 + 0 = 18
  // 5 * 1 + 2 * (-4) + 1 * 2 + 3 * 5 + 6 = 20 
  fmt::print(fg(fmt::color::violet), "{}\n", a.print());
  fmt::print(fg(fmt::color::violet), "{}\n", b.print());

};
};