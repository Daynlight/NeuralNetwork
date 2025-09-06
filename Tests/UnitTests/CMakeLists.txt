cmake_minimum_required(VERSION 3.15)

project(UnitTests LANGUAGES CXX)

set(src
  main.cpp
)

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/vendor/fmt/include/
)

add_executable(UnitTests ${src})
target_include_directories(UnitTests PRIVATE ${headers})
target_link_libraries(UnitTests PRIVATE NeuralNetwork)
target_link_libraries(UnitTests PRIVATE fmt)