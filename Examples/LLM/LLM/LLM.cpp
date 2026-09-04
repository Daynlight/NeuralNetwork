// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "LLM.h"



bool hasInput() noexcept {
  pollfd fd{
    .fd = STDIN_FILENO,
    .events = POLLIN,
    .revents = 0
  };

  return poll(&fd, 1, 0) > 0 && (fd.revents & POLLIN);
};



// =========================== //
// ======= Constructors ====== //
// =========================== //
// core
LLM::LLM::LLM() noexcept { };



LLM::LLM::~LLM() noexcept { };



// =========================== //
// ========= Control ========= //
// =========================== //
void LLM::LLM::setLearnFilePath(std::filesystem::path path) noexcept {
  if(!std::filesystem::exists(std::filesystem::path(__FILE__).parent_path() / path) || 
      std::filesystem::is_directory(std::filesystem::path(__FILE__).parent_path() / path)) return;
  path_to_learning_set = std::filesystem::path(__FILE__).parent_path() / path;
};



std::filesystem::path LLM::LLM::getLearnFilePath() const noexcept {
  return path_to_learning_set;
};



void LLM::LLM::setModelFilePath(std::filesystem::path path) noexcept {
  if(!std::filesystem::exists(std::filesystem::path(__FILE__).parent_path() / path) || 
      std::filesystem::is_directory(std::filesystem::path(__FILE__).parent_path() / path)) return;
  path_to_model_data = std::filesystem::path(__FILE__).parent_path() / path;
};



std::filesystem::path LLM::LLM::getModelFilePath() const noexcept {
  return path_to_model_data;
};



// =========================== //
// ====== Serialization ====== //
// =========================== //
void LLM::LLM::loadModelFromFile() noexcept {
  if(!std::filesystem::exists(path_to_model_data) || std::filesystem::is_directory(path_to_model_data)) return;

  std::ifstream file(path_to_model_data, std::ios::binary);
  if(!file.is_open()) return;

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string data(size, '\0');
  if(size > 0) file.read(data.data(), size);
  if(!file) return;

  model.deserialize(data);
};



void LLM::LLM::saveModelToFile() noexcept {
  std::ofstream file(path_to_model_data, std::ios::binary);
  if(!file.is_open()) return;
  std::string data = model.serialize();

  if(!data.empty()) file.write(data.data(), data.size());
};



// =========================== //
// ======= Application ======= //
// =========================== //
void LLM::LLM::learn() noexcept {
  if(!std::filesystem::exists(path_to_learning_set) || std::filesystem::is_directory(path_to_learning_set)) return;

  std::ifstream file(path_to_learning_set, std::ios::binary);
  if(!file.is_open()) return;
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  if(file_size <= context_size) return;
  std::string text(file_size, '\0');
  file.read(text.data(), file_size);
  if(!file) return;

  fmt::println(fg(fmt::color::yellow), "-- Learning");
  fmt::println(fg(fmt::color::white), "enter to abort without lossing");

  std::array<double, input_size> input;
  std::array<double, alphabet_size> target;

  model.setLearningRate(learning_rate);

  model.setActivation<1, NN::Sigmoid>();
  model.setActivation<2, NN::Sigmoid>();
  model.setActivation<3, NN::Sigmoid>();

  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      if(hasInput()){
        std::string input;
        std::getline(std::cin, input);
        fmt::println("");
        fmt::println(fg(fmt::color::yellow), "-- Learning aborted");
        saveModelToFile();
        return;
      };

      input.fill(0.0);
      target.fill(0.0);

      size_t offset = rand() % (text.size() - context_size);

      for(size_t k = 0; k < context_size; k++){
        unsigned char character = static_cast<unsigned char>(text[offset + k]);
        if(character >= alphabet_size) character = '?';
        input[k * alphabet_size + character] = 1.0;
      };

      unsigned char expected = static_cast<unsigned char>(text[offset + context_size]);

      if(expected >= alphabet_size) expected = '?';
      target[expected] = 1.0;
      NN::Utils::progressBar(i + j * learn_samples, learn_samples * epoch);

      model.setInput(input);
      model.forward();
      model.backprop(target);
    };
  };

  saveModelToFile();
};



std::string LLM::LLM::getRespond(const std::string &message) noexcept {
  std::string context = message;
  std::string response = "";

  std::array<double, input_size> input;

  for(size_t i = 0; i < response_size; i++){
    input.fill(0.0);

    size_t characters = std::min(context.size(), context_size);
    size_t padding = context_size - characters;
    size_t start = context.size() - characters;

    for(size_t k = 0; k < padding; k++)
      input[k * alphabet_size + ' '] = 1.0;

    for(size_t k = 0; k < characters; k++){
      unsigned char character = static_cast<unsigned char>(context[start + k]);
      if(character >= alphabet_size) character = '?';
      input[(padding + k) * alphabet_size + character] = 1.0;
    };

    model.setInput(input);
    model.forward();

    double* result = model.getResult();
    unsigned char predicted = 0;

    for(unsigned int j = 1; j < alphabet_size; j++)
      if(result[j] > result[predicted]) predicted = j;

    response += static_cast<char>(predicted);
    context += static_cast<char>(predicted);
  };

  return response;
};



void LLM::LLM::printHelp() const noexcept {
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "Welcome to LLM");
  fmt::println(fg(fmt::color::blue), "help - see help message");
  fmt::println(fg(fmt::color::blue), "learn - to start learning from file");
  fmt::println(fg(fmt::color::blue), "set_learn_file - to set learn file");
  fmt::println(fg(fmt::color::blue), "quit - exit");
};



void LLM::LLM::onStart() noexcept {
  loadModelFromFile();
  printHelp();
};



void LLM::LLM::onUpdate() noexcept {
  std::string input = "";
  fmt::print(fg(fmt::color::white), "> ");
  std::getline(std::cin, input);

  if(input == "quit") { 
    is_running = false; 
    return; 
  };
  if(input == "help") {
    printHelp();
    return;
  };
  if(input == "learn") {
    learn();
    return;
  };
  if(input == "set_learn_file") {
    std::string file_name = "";
    fmt::print(fg(fmt::color::white), "file_name > ");
    std::getline(std::cin, file_name);
    
    if(!std::filesystem::exists(std::filesystem::path(__FILE__).parent_path() / std::filesystem::path(file_name)) || 
        std::filesystem::is_directory(std::filesystem::path(__FILE__).parent_path() / std::filesystem::path(file_name))){
      fmt::println(fg(fmt::color::red) | fmt::emphasis::bold, "{} didn't exist", file_name); 
      return;
    };
  
    setLearnFilePath(file_name);
    fmt::println(fg(fmt::color::green) | fmt::emphasis::bold, "Current Learn File: {}", getLearnFilePath().string()); 
    return;
  };

  std::string res = getRespond(input);
  fmt::println(fg(fmt::color::azure), "{}", res);
  fmt::println("size = {}", res.size());
};



void LLM::LLM::onEnd() noexcept {
  saveModelToFile();
};



void LLM::LLM::application() noexcept {
  onStart();
  while(is_running) onUpdate();
  onEnd();
};
