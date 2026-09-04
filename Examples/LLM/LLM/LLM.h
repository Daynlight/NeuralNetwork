// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <fmt/base.h>
#include <fmt/color.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <poll.h>
#include <unistd.h>

#include <NeuralNetwork/NeuralNetwork.h>
#include <NeuralNetwork/Utils.h>



namespace LLM{
class LLM{
// ======================================== //
// ================= Data ================= //
// ======================================== //
private:
  NN::NeuralNetwork<4096, 512, 256, 128, 1> model;
  std::filesystem::path path_to_learning_set = std::filesystem::path(__FILE__).parent_path() / "../data/learn_set.txt";
  std::filesystem::path path_to_model_data = std::filesystem::path(__FILE__).parent_path() / "../data/model";
  bool model_is_loaded_from_file = false;
  bool is_running = true;

  const double learning_rate = 0.001;
  const unsigned int learn_samples = 1000;
  const unsigned int epoch = 100;
  static constexpr size_t context_size = 32;
  static constexpr size_t alphabet_size = 128;
  static constexpr size_t input_size = context_size * alphabet_size;
  static constexpr size_t response_size = 256;



// ======================================== //
// ============== Functions =============== //
// ======================================== //
// =========================== //
// ======= Constructors ====== //
// =========================== //
public:
  LLM() noexcept;
  ~LLM() noexcept;

// =========================== //
// ========= Control ========= //
// =========================== //
public:
  void setLearnFilePath(std::filesystem::path path) noexcept;
  std::filesystem::path getLearnFilePath() const noexcept;
  void setModelFilePath(std::filesystem::path path) noexcept;
  std::filesystem::path getModelFilePath() const noexcept;
  
// =========================== //
// ====== Serialization ====== //
// =========================== //
public:
  void loadModelFromFile() noexcept;
  void saveModelToFile() noexcept;

// =========================== //
// ======= Application ======= //
// =========================== //
public:
  void learn() noexcept;
  std::string getRespond(const std::string& message) noexcept;
  
  void printHelp() const noexcept;

  void onStart() noexcept;
  void onUpdate() noexcept;
  void onEnd() noexcept;
  void application() noexcept;
};
};