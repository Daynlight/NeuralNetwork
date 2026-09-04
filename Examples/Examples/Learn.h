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
#include "NeuralNetwork/Utils.h"



namespace ExamplesTemplatedCPU::Learn{
void Minus(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "---------- LearnMinus ---------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");

  std::srand(std::time(nullptr));
  NN::Layer<2, 1> e;
  NN::Layer<1, 0> g;

  const double learning_rate = 0.001;
  const unsigned int modulo_number = 100;
  const unsigned int learn_samples = 1000;
  const unsigned int tests = 1000000;
  const unsigned int epoch = 1000;
  const double tolerance = 1.0;
  e.setLearningRate(learning_rate);
  g.setLearningRate(learning_rate);
  fmt::println(fg(fmt::color::yellow), "-- Parameters:");
  fmt::println(fg(fmt::color::white), "learning_rate = {}", learning_rate);
  fmt::println(fg(fmt::color::white), "modulo_number = {}", modulo_number);
  fmt::println(fg(fmt::color::white), "learn_samples = {}", learn_samples);
  fmt::println(fg(fmt::color::white), "tests = {}", tests);
  fmt::println(fg(fmt::color::white), "epoch = {}", epoch);
  fmt::println(fg(fmt::color::white), "tolerance = {}", tolerance);

  fmt::println(fg(fmt::color::yellow), "-- Learning from random set");
  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      double x = (rand()%modulo_number);
      double y = (rand()%modulo_number);

      NN::Utils::progressBar(i + j * learn_samples, learn_samples * epoch);

      e.setNodes({x / modulo_number, y / modulo_number});
      e.forward(g);

      double val = x - y;
      double res = (val / modulo_number);
      g.backprop_initial(e, {res});
      e.backprop(g);
    };
  };

  fmt::println(fg(fmt::color::yellow), "-- Testing on random set");
  double sum = 0;
  for(unsigned int i = 0; i < tests; i++) {
    double x = (rand()%modulo_number);
    double y = (rand()%modulo_number);

    NN::Utils::progressBar(i, tests);

    e.setNodes({x / modulo_number, y / modulo_number});
    e.forward(g);

    double val = x - y;
    double res = val / modulo_number;
    if ((fabs(g[0] - res) * modulo_number < tolerance ? 1 : 0)) sum += 1;
  };
  if((sum / tests) * 100 >= 90) fmt::println(fg(fmt::color::green), "avg: {}%", (sum / tests) * 100);
  else                          fmt::println(fg(fmt::color::red), "avg: {}%", (sum / tests) * 100);


  fmt::println(fg(fmt::color::yellow), "-- Test on defined values");
  double x = 100.0;
  double y = 20.0;
  e.setNodes({x / modulo_number, y / modulo_number});
  e.forward(g);
  fmt::println(fg(fmt::color::violet), "{} - {} = {}", x, y, g[0] * modulo_number);
};



void XOR(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "---------- LearnXOR -----------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");

  std::srand(std::time(nullptr));
  NN::NeuralNetwork network = NN::NeuralNetwork<2, 2, 1>();

  const double learning_rate = 0.01;
  const unsigned int modulo_number = 2;
  const unsigned int learn_samples = 2000;
  const unsigned int epoch = 1000;
  const unsigned int tests = 10000;
  network.setLearningRate(learning_rate);
  network.setActivation<1>(NN::ActivationType::SIGMOIDTYPE);
  fmt::println(fg(fmt::color::yellow), "-- Parameters:");
  fmt::println(fg(fmt::color::white), "learning_rate = {}", learning_rate);
  fmt::println(fg(fmt::color::white), "modulo_number = {}", modulo_number);
  fmt::println(fg(fmt::color::white), "learn_samples = {}", learn_samples);
  fmt::println(fg(fmt::color::white), "tests = {}", tests);
  fmt::println(fg(fmt::color::white), "epoch = {}", epoch);

  fmt::println(fg(fmt::color::yellow), "-- Learning from random set");
  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      double x = (rand()%modulo_number);
      double y = (rand()%modulo_number);

      NN::Utils::progressBar(i + j * learn_samples, learn_samples * epoch);

      network.setInput({x, y});
      network.forward();
      double res = ((x != 0) != (y != 0));
      network.backprop({res});
    };
  };

  fmt::println(fg(fmt::color::yellow), "-- Testing on random set");
  double sum = 0;
  for(unsigned int i = 0; i < tests; i++) {
    double x = (rand()%modulo_number);
    double y = (rand()%modulo_number);

    NN::Utils::progressBar(i, tests);

    network.setInput({x, y});
    network.forward();
    double res = ((x != 0) != (y != 0));
    if ((network.getResult()[0] >= 0.5) == res) sum += 1;
  };
  if((sum / tests) * 100 >= 90) fmt::println(fg(fmt::color::green), "avg: {}%", (sum / tests) * 100);
  else                          fmt::println(fg(fmt::color::red), "avg: {}%", (sum / tests) * 100);


  fmt::println(fg(fmt::color::yellow), "-- Test on defined values");
  double x = 1.0;
  double y = 0.0;
  network.setInput({x, y});
  network.forward();
  fmt::println(fg(fmt::color::violet), "{} XOR {} = {}", x, y, network.getResult()[0]);
};



void Func(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "---------- LearnFunc ----------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");

  std::srand(std::time(nullptr));
  NN::NeuralNetwork network = NN::NeuralNetwork<3, 8, 8, 1>();

  std::function<double(double, double, double)> fun =
  [](double x, double y, double z){
    return x * y + z;
  };

  const double learning_rate = 0.0001;
  const unsigned int modulo_number = 5;
  const unsigned int learn_samples = 200;
  const unsigned int epoch = 1000;
  const unsigned int tests = 10000;
  const double tolerance = 1.0f;
  network.setLearningRate(learning_rate);
  fmt::println(fg(fmt::color::yellow), "-- Parameters:");
  fmt::println(fg(fmt::color::white), "learning_rate = {}", learning_rate);
  fmt::println(fg(fmt::color::white), "modulo_number = {}", modulo_number);
  fmt::println(fg(fmt::color::white), "learn_samples = {}", learn_samples);
  fmt::println(fg(fmt::color::white), "tests = {}", tests);
  fmt::println(fg(fmt::color::white), "epoch = {}", epoch);
  fmt::println(fg(fmt::color::white), "tolerance = {}", tolerance);

  fmt::println(fg(fmt::color::yellow), "-- Learning from random set");
  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      double x = (rand()%modulo_number);
      double y = (rand()%modulo_number);
      double z = (rand()%modulo_number);

      NN::Utils::progressBar(i + j * learn_samples, learn_samples * epoch);

      network.setInput({x / modulo_number, y / modulo_number, z / modulo_number});
      network.forward();
      double res = fun(x, y, z) / modulo_number;
      network.backprop({res});
    };
  };

  fmt::println(fg(fmt::color::yellow), "-- Testing on random set");
  double sum = 0;
  for(unsigned int i = 0; i < tests; i++) {
    double x = (rand()%modulo_number);
    double y = (rand()%modulo_number);
    double z = (rand()%modulo_number);

    NN::Utils::progressBar(i, tests);

    network.setInput({x / modulo_number, y / modulo_number, z / modulo_number});
    network.forward();
    double res = fun(x, y, z) / modulo_number;
    if ((fabs(network.getResult()[0] - res) < tolerance ? 1 : 0)) sum += 1;
  };
  if((sum / tests) * 100 >= 90) fmt::println(fg(fmt::color::green), "avg: {}%", (sum / tests) * 100);
  else                          fmt::println(fg(fmt::color::red), "avg: {}%", (sum / tests) * 100);


  fmt::println(fg(fmt::color::yellow), "-- Test on defined values");
  double x = 1.0;
  double y = 2.0;
  double z = 4.0;
  network.setInput({x / modulo_number, y / modulo_number, z / modulo_number});
  network.forward();
  fmt::println(fg(fmt::color::violet), "{} * {} + {} = {}", x, y, z, network.getResult()[0] * modulo_number);
};
};