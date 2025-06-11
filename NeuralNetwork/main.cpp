#include "Exampe.h"
#include "UnitTests.h"
#include "Node.hpp"
#include <cstdio>

int main(){
  UnitTests::runAllTests();
  Examples::NodeExample();
  Examples::LayerExample();


  return 0;
}