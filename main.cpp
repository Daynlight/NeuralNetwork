#include "Tests/Unit/unit_tests.h"
#include "Tests/Integration/integration_tests.h"
#include "Tests/Regression/regression_tests.h"
#include "Tests/Performance/performance_tests.h"
#include "Tests/Security/security_tests.h"
#include "Tests/Stress/stress_tests.h"
#include "Tests/Soak/soak_tests.h"

#include "Examples/Essentials/Vector/vector.h"

int main(){
  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "======== Tests ========\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");
  UnitTests::AllTests();
  IntegrationTests::AllTests();
  RegressionTests::AllTests();
  PerformanceTests::AllTests();
  SecurityTests::AllTests();
  StressTests::AllTests();
  SoakTests::AllTests();

  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "====== Examples ======\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "======================\n");
  Examples::Vector();

  return 0;
}