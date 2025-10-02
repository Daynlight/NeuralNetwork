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
  layer.setActivation(NN::ActivationType::LINEARTYPE);
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
  NN::Layer<2, 2> h;
  NN::Layer<2, 0> d;
  c.setNodes({2, 1});
  c.forward(h);
  h.forward(d);
  d.backprop_initial(h, {3, 3, 1});
  h.backprop(d);
  c.backprop(h);
  fmt::print(fg(fmt::color::violet), "{}\n", c.print());
  fmt::print(fg(fmt::color::violet), "{}\n", h.print());
  fmt::print(fg(fmt::color::violet), "{}\n", d.print());

  // xor learn
  fmt::print(fg(fmt::color::crimson), "xor learn\n");
  std::srand(std::time(nullptr));
  NN::Layer<2, 2> e;
  NN::Layer<2, 1> f;
  NN::Layer<1, 0> g;

  const double learning_rate = 0.1;
  const unsigned int modulo_number = 2;
  const unsigned int learn_samples = 500000;
  const unsigned int tests = 10000;
  const double tolerance = 0.1;

  e.setLearningRate(learning_rate);
  f.setLearningRate(learning_rate);
  g.setLearningRate(learning_rate);

  e.setActivation(NN::ActivationType::SIGMOIDTYPE);
  f.setActivation(NN::ActivationType::SIGMOIDTYPE);
  
  for(unsigned int i = 0; i < learn_samples; i++) {
    double x = (rand()%modulo_number);
    double y = (rand()%modulo_number);
    e.setNodes({x, y});
    e.forward(f);
    f.forward(g);
    double res = x != y;
    g.backprop_initial(f, {res});
    f.backprop(g);
    e.backprop(f);
  }

  double avg = 0;
  for(unsigned int i = 0; i < tests; i++) {
    double x = (rand()%modulo_number);
    double y = (rand()%modulo_number);
    e.setNodes({x, y});
    e.forward(f);
    f.forward(g);
    double res = x != y;
    if ((g[0] > 0.5 ? 1 : 0) == res) avg += 1;
  }


  fmt::print(fg(fmt::color::violet), "{}\n", e.print());
  fmt::print(fg(fmt::color::violet), "{}\n", f.print());
  fmt::print(fg(fmt::color::violet), "{}\n", g.print());
  fmt::print(fg(fmt::color::red), "avg: {}%\n", (avg / tests) * 100);

  e.setNodes({0, 0});
  e.forward(f);
  f.forward(g);
  fmt::print(fg(fmt::color::red), "0, 0: -> {}\n", g[0] > 0.5 ? 1 : 0);

  e.setNodes({0, 1});
  e.forward(f);
  f.forward(g);
  fmt::print(fg(fmt::color::red), "0, 1: -> {}\n", g[0] > 0.5 ? 1 : 0);

  e.setNodes({1, 0});
  e.forward(f);
  f.forward(g);
  fmt::print(fg(fmt::color::red), "1, 0 -> {}\n", g[0] > 0.5 ? 1 : 0);

  e.setNodes({1, 1});
  e.forward(f);
  f.forward(g);
  fmt::print(fg(fmt::color::red), "1, 1 -> {}\n", g[0] > 0.5 ? 1 : 0);
};
};