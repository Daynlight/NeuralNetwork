cmake_minimum_required(VERSION 3.15)

############################## Set Default Path ##############################
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/output)

######################### Set Project Language #########################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

########################### add submodules ########################### 
add_subdirectory(vendor/fmt)

########################### add subdirectories ########################### 
add_subdirectory(NeuralNetwork)

add_subdirectory(Examples)
add_subdirectory(Tests)