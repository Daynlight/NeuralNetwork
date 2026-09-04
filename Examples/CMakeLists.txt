# Neural Network
# Copyright 2026 Daynlight
# Licensed under the GNU General.
# See LICENSE file for details.



cmake_minimum_required(VERSION 3.10)

########################### Example Project ########################### 
project(Example LANGUAGES CXX)

find_package(fmt CONFIG REQUIRED)

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/vendor/fmt/include/
)

set(src 
  main.cpp
)

add_executable(Example ${src})

target_include_directories(Example PRIVATE ${headers})
target_link_libraries(Example PRIVATE 
  NeuralNetwork::NeuralNetwork
  fmt::fmt
)
