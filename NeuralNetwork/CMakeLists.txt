cmake_minimum_required(VERSION 3.15)

project(NeuralNetwork LANGUAGES CXX)

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(src
  Layer/Layer.hpp
  NeuralNetwork.cpp
)

add_library(NeuralNetwork STATIC ${headers} ${src})

target_include_directories(NeuralNetwork PUBLIC ${headers})