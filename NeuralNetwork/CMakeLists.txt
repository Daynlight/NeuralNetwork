cmake_minimum_required(VERSION 3.20)

project(NeuralNetwork LANGUAGES CXX)

# Use C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect source files
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${CMAKE_SOURCE_DIR}/UnitTests.h
    ${CMAKE_SOURCE_DIR}/Example.h
    ${CMAKE_SOURCE_DIR}/NeuralNetwork.h
    ${CMAKE_SOURCE_DIR}/Node.hpp
    ${CMAKE_SOURCE_DIR}/Layer.hpp
)

# Create executable
add_executable(NeuralNetwork ${SRC_FILES})
