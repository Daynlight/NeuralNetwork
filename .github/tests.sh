#!/bin/bash

mkdir build
cmake . --build build/ --config Debug
../build/output/Tests.exe