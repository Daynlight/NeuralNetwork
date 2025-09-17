#pragma once
#include "Essentials/Vector/vector.h"


namespace UnitTests {
void AllTests() {
  int total=0, passed=0;
#define RUN(f) total++; if(f) passed++;
  RUN(Vector());

#undef RUN
  if(passed == total) 
    fmt::print(fg(fmt::color::sky_blue), "\n=== Test Score: {:d}\\{:d} passed ===\n", passed, total);
  else 
    fmt::print(fg(fmt::color::dark_orange), "\n=== Test Score: {:d}\\{:d} passed ===\n", passed, total);
};
};