#pragma once

#include <string.h>

#include "fmt/base.h"
#include "fmt/color.h"


namespace UnitTests{
class Assert{
private:
  unsigned int total = 0;
  unsigned int passed = 0;

public:
  template<typename T>
  void eq(const std::string& message, const T& a, const T& b){
    total++;
    if(a != b){
      fmt::print(fg(fmt::color::dark_golden_rod) , "%s\n", message.c_str());
      return;
    };

    passed++;
  };

};
};