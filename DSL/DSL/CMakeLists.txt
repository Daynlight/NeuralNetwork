cmake_minimum_required(VERSION 3.10)

project(DSL LANGUAGES C)

set(src
  vector/vector.c
)

set(header
  dsl.h
  vector/vector.h
)


add_library(DSL STATIC ${header} ${src})