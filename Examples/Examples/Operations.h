// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include "fmt/base.h"
#include "fmt/color.h"

#include <iomanip>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <functional>

#include "NeuralNetwork/Layer/Layer.h"
#include "NeuralNetwork/NeuralNetwork.h"



namespace ExamplesTemplatedCPU::Operations{
void BaseOperations(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "---- BaseOperationExamples ----");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");
  
  fmt::println(fg(fmt::color::yellow), "-- Create Layer");
  NN::Layer<2, 1> layer;
  fmt::println(fg(fmt::color::white), "{}", layer.print());
  
  fmt::println(fg(fmt::color::yellow), "-- Set Nodes");
  layer.setNodes({6});
  fmt::println(fg(fmt::color::white), "{}", layer.print());
  
  fmt::println(fg(fmt::color::yellow), "-- Set Weights");
  layer.setWeights({2, 4});
  fmt::println(fg(fmt::color::white), "{}", layer.print());

  fmt::println(fg(fmt::color::yellow), "-- Set Loss");
  layer.setLoss<NN::MSE>();
  fmt::println(fg(fmt::color::white), "{}", layer.print());

  fmt::println(fg(fmt::color::yellow), "-- Set Activation");
  fmt::println(fg(fmt::color::green), "---- SIGMOID");
  layer.setActivation<NN::Sigmoid>();
  fmt::println(fg(fmt::color::white), "{}", layer.print());

  fmt::println(fg(fmt::color::green), "---- LINEAR");
  layer.setActivation<NN::Linear>();
  fmt::println(fg(fmt::color::white), "{}", layer.print());
};



void Forward(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "----------- Forward -----------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");

  NN::Layer<4, 3> a;
  fmt::println(fg(fmt::color::purple), "a => {}", a.print());
  NN::Layer<3, 0> b;
  fmt::println(fg(fmt::color::pink), "b => {}", b.print());

  a.setNodes({5, 2, 1, 3});
  a.setWeights({1, 2 ,1, -1, 5,   2, 1, 3, 1, 0,   1, -4, 2, 5, 6});
  fmt::println(fg(fmt::color::purple), "a initialize => {}", a.print());
  
  a.forward(b);
  fmt::println(fg(fmt::color::yellow), "a => forward => b");
  fmt::println(fg(fmt::color::white), "5 * 1 + 2 * 2 + 1 * 1 + 3 * (-1) + 5 = 12");
  fmt::println(fg(fmt::color::white), "5 * 2 + 2 * 1 + 1 * 3 + 3 * 1 + 0 = 18");
  fmt::println(fg(fmt::color::white), "5 * 1 + 2 * (-4) + 1 * 2 + 3 * 5 + 6 = 20"); 

  fmt::println(fg(fmt::color::purple), "a => {}", a.print());
  fmt::println(fg(fmt::color::pink), "b => {}", b.print());
};



void Backprop(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "----------- Backprop ----------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");

  NN::Layer<2, 2> c;
  NN::Layer<2, 2> h;
  NN::Layer<2, 0> d;
  c.setWeights({1,1,1,  1,1,1});
  h.setWeights({1,1,1,   1,1,1});
  c.setNodes({2, 1});

  fmt::println(fg(fmt::color::pink), "c => {}", c.print());
  fmt::println(fg(fmt::color::purple), "h => {}", h.print());
  fmt::println(fg(fmt::color::violet), "d => {}", d.print());
  
  
  c.forward(h);
  h.forward(d);
  fmt::println(fg(fmt::color::yellow), "c => forward => h => forward => d");

  fmt::println(fg(fmt::color::pink), "c => {}", c.print());
  fmt::println(fg(fmt::color::purple), "h => {}", h.print());
  fmt::println(fg(fmt::color::violet), "d => {}", d.print());

  d.backprop_initial({3, 3, 1});
  h.backprop(d);
  c.backprop(h);
  fmt::println(fg(fmt::color::yellow), "c <= backprop <= h <= backprop <= d");

  fmt::println(fg(fmt::color::pink), "c => {}", c.print());
  fmt::println(fg(fmt::color::purple), "h => {}", h.print());
  fmt::println(fg(fmt::color::violet), "d => {}", d.print());
};
};