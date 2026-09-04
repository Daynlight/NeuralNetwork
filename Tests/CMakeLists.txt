# Neural Network
# Copyright 2026 Daynlight
# Licensed under the GNU General v3.
# See LICENSE file for details.



cmake_minimum_required(VERSION 3.15)

find_package(GTest CONFIG REQUIRED)



set(tests_src
  Unit/CPUTemplated/Activation.cpp
  Unit/CPUTemplated/Layer.cpp
  Unit/CPUTemplated/Loss.cpp
  Unit/CPUTemplated/NeuralNetwork.cpp
)



add_executable(unit_tests ${tests_src})

target_link_libraries(unit_tests PRIVATE
  NeuralNetwork::NeuralNetwork
  GTest::gtest
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)