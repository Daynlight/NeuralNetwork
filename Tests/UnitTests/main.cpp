#include "Assert.h"
#include "UnitTests.h"


int main(){
  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "====== UnitTests ======\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");

  UnitTests::Assert assert;


  return 0;
};