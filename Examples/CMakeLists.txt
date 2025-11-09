cmake_minimum_required(VERSION 3.10)

############################## Set Default Path ##############################
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/output)

######################### Set Project Language #########################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


########################### Example Project ########################### 
project(Example LANGUAGES CXX)
set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/vendor/fmt/include/
)

set(src 
  main.cpp
)

add_executable(Example ${src} ${headers})
target_include_directories(Example PRIVATE ${headers})
target_link_libraries(Example PRIVATE NeuralNetwork)
target_link_libraries(Example PRIVATE fmt)