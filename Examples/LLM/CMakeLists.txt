# Neural Network
# Copyright 2026 Daynlight
# Licensed under the GNU General.
# See LICENSE file for details.



cmake_minimum_required(VERSION 3.10)

########################### LLM Project ########################### 
project(LLM LANGUAGES CXX)

find_package(fmt CONFIG REQUIRED)

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/
)

set(src 
  main.cpp
  LLM/LLM.cpp
)

add_executable(LLM ${src})

target_include_directories(LLM PRIVATE ${headers})
target_link_libraries(LLM PRIVATE 
  NeuralNetwork::NeuralNetwork
  fmt::fmt
)

