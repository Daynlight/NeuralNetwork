# Makefile for NeuralNetwork project

CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -I. -I./NeuralNetwork
SRC := main.cpp
OUTDIR := build
OUT := $(OUTDIR)/NeuralNetwork.exe

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT)

run: $(OUT)
	./$(OUT)

clean:
	rm -rf $(OUTDIR)
