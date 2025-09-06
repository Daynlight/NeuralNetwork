cmake_minimum_required(VERSION 3.10)

project(Tests LANGUAGES C)

set(src 
  test.c
)

add_executable(Tests ${src})

target_link_libraries(Tests DSL)