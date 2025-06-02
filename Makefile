# Makefile for NeuralNetwork Project

CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -I./NeuralNetwork
SRC := $(wildcard NeuralNetwork/*.cpp)
OUTDIR := build
OUT := $(OUTDIR)/NeuralNetwork.exe

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(SRC) main.cpp -o $(OUT)

run: $(OUT)
	./$(OUT)

clean:
	rm -rf $(OUTDIR)
