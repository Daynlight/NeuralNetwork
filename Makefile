# Makefile to build NeuralNetwork

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
SRC = src/Node.cpp tests/example.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = build/NeuralNetwork.exe

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -rf build
