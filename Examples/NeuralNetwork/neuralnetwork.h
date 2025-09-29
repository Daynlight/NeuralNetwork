#pragma once
#include "dsl.h"
#include <cstdlib>
#include <ctime>

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

  // backprop
  fmt::print(fg(fmt::color::crimson), "backprop\n");
  NN::Layer<2, 2> c;
  NN::Layer<2, 0> d;
  c.setNodes({2, 1});
  c.setWeights({1, 1, 1,   1, 1, 1});
  c.forward(d);
  fmt::print(fg(fmt::color::violet), "{}\n", c.print());
  fmt::print(fg(fmt::color::violet), "{}\n", d.print());
  d.backprop_initial(c, {3, 3, 1});
  fmt::print(fg(fmt::color::violet), "{}\n", c.print());
  fmt::print(fg(fmt::color::violet), "{}\n", d.print());

  // sum learn
  fmt::print(fg(fmt::color::crimson), "sum\n");
  std::srand(std::time(nullptr));
  NN::Layer<2, 1> e;
  NN::Layer<1, 0> f;
  for(unsigned int i = 0; i < 1000000; i++) {
    double x = rand()%10;
    double y = rand()%10;
    e.setNodes({x, y});
    e.forward(f);
    double res = x + y;
    f.backprop_initial(e, {res});
  }

  double avg = 0;
  for(unsigned int i = 0; i < 100; i++) {
    double x = rand()%2;
    double y = rand()%2;
    e.setNodes({x, y});
    e.forward(f);
    double res = x && y;
    if(f[1] == res) avg += 1;
  }

  avg /= 100;
  fmt::print(fg(fmt::color::red), "avg: {}\n", avg);
  fmt::print(fg(fmt::color::violet), "{}\n", e.print());
  fmt::print(fg(fmt::color::violet), "{}\n", f.print());
};
};