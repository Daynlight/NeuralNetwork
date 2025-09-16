cmake_minimum_required(VERSION 3.10)

project(DSL LANGUAGES CXX)

set(headers
  dsl.h
  Vector/vector.h
  Vector/vector.hpp
  Math/ModuloZ.h
  Math/ModuloZ.hpp
)

set(src
  dsl.cpp
)

add_library(DSL STATIC ${headers} ${src})

target_include_directories(DSL PUBLIC ${headers})