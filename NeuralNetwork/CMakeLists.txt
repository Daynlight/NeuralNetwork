cmake_minimum_required(VERSION 3.15)

project(NeuralNetwork LANGUAGES CXX)


set(src
  Layer/Layer.hpp
  NeuralNetwork.cpp
)

add_library(NeuralNetwork STATIC ${src})
add_library(NeuralNetwork::NeuralNetwork ALIAS NeuralNetwork)

target_include_directories(
  NeuralNetwork
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)