#include "Examples/Essentials/Vector/vector.h"
#include "Examples/NeuralNetwork/neuralnetwork.h"

int main(){
  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "====== Examples ======\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "======================\n");
  Examples::Vector();

  Examples::NeuralNetwork();

  return 0;
}