#pragma once
#include "fmt/base.h"
#include "fmt/color.h"

#include <iomanip>
#include <iostream>
#include <ctime>
#include <cstdlib>

#include "Layer/Layer.h"
#include "../NeuralNetwork/NeuralNetwork.h"



namespace Examples{
void BaseOperations(){
  fmt::print(fg(fmt::color::aquamarine) , "### BaseOperations\n");
  
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
}

void Forward(){
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
}

void Backprop(){
  // backprop
  fmt::print(fg(fmt::color::crimson), "backprop\n");
  NN::Layer<2, 2> c;
  NN::Layer<2, 2> h;
  NN::Layer<2, 0> d;
  c.setWeights({1,1,1,  1,1,1});
  h.setWeights({1,1,1,   1,1,1});
  c.setNodes({2, 1});
  c.forward(h);
  h.forward(d);
  d.backprop_initial(h, {3, 3, 1});
  h.backprop(d);
  c.backprop(h);
  fmt::print(fg(fmt::color::violet), "{}\n", c.print());
  fmt::print(fg(fmt::color::violet), "{}\n", h.print());
  fmt::print(fg(fmt::color::violet), "{}\n", d.print());
};



void LearnMinus(){
  // minus learn
  fmt::print(fg(fmt::color::crimson), "minus learn\n");
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

  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      double x = (rand()%modulo_number);
      double y = (rand()%modulo_number);

      if ((i + j * learn_samples) % 100 == 0 || (i + j * learn_samples) == learn_samples * epoch - 1) {
        float progress = (float)(i + j * learn_samples) / (learn_samples * epoch);
        int barWidth = 50;

        std::cout << "\r[";
        int pos = barWidth * progress;

        for (int k = 0; k < barWidth; k++) {
          if (k < pos) std::cout << "=";
          else if (k == pos) std::cout << ">";
          else std::cout << " ";
        };

        std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << "%";
        std::cout.flush();
      };

      e.setNodes({x / modulo_number, y / modulo_number});
      e.forward(g);
      double res = ((x - y) / modulo_number);
      g.backprop_initial(e, {res});
      e.backprop(g);
    };
  };

  printf("\n");
  double avg = 0;
  for(unsigned int i = 0; i < tests; i++) {
    double x = (rand()%modulo_number);
    double y = (rand()%modulo_number);
    e.setNodes({x / modulo_number, y / modulo_number});
    e.forward(g);
    double res = (x - y) / modulo_number;
    if ((fabs(g[0] - res) * modulo_number < tolerance ? 1 : 0)) avg += 1;
  }

  fmt::print(fg(fmt::color::red), "avg: {}%\n", (avg / tests) * 100);
  double x = 100.0;
  double y = 20.0;

  e.setNodes({x / modulo_number, y / modulo_number});
  e.forward(g);
  fmt::print(fg(fmt::color::violet), "{} - {} = {}\n", x, y, g[0] * modulo_number);
};



double fun(double x, double y, double z){
  return x * y + z;
};

void learnFunction(){
  fmt::print(fg(fmt::color::crimson), "function learn\n");

  std::srand(std::time(nullptr));

  const double learning_rate = 0.0001;
  const unsigned int modulo_number = 5;
  const unsigned int learn_samples = 200;
  const unsigned int epoch = 1000;
  const unsigned int tests = 10000;
  const double tolerance = 1.0f;

  NN::NeuralNetwork network = NN::NeuralNetwork<3, 8, 8, 1>();
  network.setLearningRate(learning_rate);

  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      double x = (rand()%modulo_number);
      double y = (rand()%modulo_number);
      double z = (rand()%modulo_number);

      if ((i + j * learn_samples) % 100 == 0 || (i + j * learn_samples) == learn_samples * epoch - 1) {
        float progress = (float)(i + j * learn_samples) / (learn_samples * epoch);
        int barWidth = 50;

        std::cout << "\r[";
        int pos = barWidth * progress;

        for (int k = 0; k < barWidth; k++) {
          if (k < pos) std::cout << "=";
          else if (k == pos) std::cout << ">";
          else std::cout << " ";
        };

        std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << "%";

        std::cout.flush();
      };
      
      network.setNodes({x / modulo_number, y / modulo_number, z / modulo_number});
      network.forward();
      double res = fun(x, y, z) / modulo_number;
      network.backprop({res});
    };
  };

  printf("\n");
  double avg = 0;
  for(unsigned int i = 0; i < tests; i++) {
    double x = (rand()%modulo_number);
    double y = (rand()%modulo_number);
    double z = (rand()%modulo_number);
    network.setNodes({x / modulo_number, y / modulo_number, z / modulo_number});
    network.forward();
    double res = fun(x, y, z) / modulo_number;
    if ((fabs(network.getResult()[0] - res) < tolerance ? 1 : 0)) avg += 1;
  };

  fmt::print(fg(fmt::color::red), "avg: {}%\n", (avg / tests) * 100);
  double x = 1.0;
  double y = 2.0;
  double z = 4.0;

  network.setNodes({x / modulo_number, y / modulo_number, z / modulo_number});
  network.forward();
  fmt::print(fg(fmt::color::violet), "{} * {} + {} = {}\n", x, y, z, network.getResult()[0] * modulo_number);
};



void NeuralNetwork() {
  fmt::print(fg(fmt::color::aquamarine) , "=== NeuralNetwork Example ===\n");
  LearnMinus();
};



void Examples(){
  // BaseOperations();
  LearnMinus();
  learnFunction();
};
};