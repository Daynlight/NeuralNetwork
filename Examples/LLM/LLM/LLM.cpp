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



void LLM::LLM::setLearningRate(double value) noexcept {
  if(value < 0.00000001) return;
  learning_rate = value;
};



void LLM::LLM::setLearningEpoch(int value) noexcept {
  if(value < 0) return;
  epoch = value;
};



void LLM::LLM::setLearningSamples(int value) noexcept {
  if(value < 0) return;
  learn_samples = value;
};



void LLM::LLM::setLearningSetRepeats(int value) noexcept {
  if(value < 0) return;
  learning_set_repeats = value;
};



void LLM::LLM::setAdditionalAcuracyShow(int value) noexcept {
  if(value <= 0) additional_acuracy_show = false;
  else  additional_acuracy_show = true;
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
  if(!std::filesystem::exists(path_to_model_data) || std::filesystem::is_directory(path_to_model_data)){
    model.setWeights<0>(-0.05, 0.05);
    model.setWeights<1>(-0.1, 0.1);
    model.setWeights<2>(-0.1, 0.1);
    return;
  };
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
// ========= Helpers ========= //
// =========================== //
std::string LLM::LLM::readLearningSet() const noexcept {
  if(!std::filesystem::exists(path_to_learning_set) || std::filesystem::is_directory(path_to_learning_set)) return "";

  std::ifstream file(path_to_learning_set, std::ios::binary);
  if(!file.is_open()) return "";
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  if(file_size <= context_size) return "";

  std::string text(file_size, '\0');
  file.read(text.data(), file_size);
  if(!file) return "";

  return text;
};



std::string LLM::LLM::prepareLearningSet(const std::string& text) const noexcept {
  std::stringstream stream(text);

  std::string line;
  std::string compact_text;

  while(std::getline(stream, line)){
    if(line.empty()) continue;
    while(!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
      line.pop_back();
    if(line.empty()) continue;
    if(line.back() == ':') continue;
    if(!compact_text.empty())
      compact_text += ' ';
    compact_text += line;
  };

  std::string normalized_text;
  normalized_text.reserve(compact_text.size());
  bool last_was_space = false;

  for(unsigned char character : compact_text){
    if(std::isspace(character)){
      if(!last_was_space)
        normalized_text += ' ';
      last_was_space = true;
    }else{
      normalized_text += static_cast<char>(character);
      last_was_space = false;
    };
  };

  return normalized_text;
};



template<size_t S>
std::pair<double, std::array<double, S>> LLM::LLM::calculateLearningSetEntropy(const std::string &text) const noexcept {
  std::array<double, S> frequency{};
  size_t total = 0;

  for(unsigned char c : text){
    if(c >= S) c = '?';
    frequency[c]++;
    total++;
  };

  double entropy = 0.0;
  for(size_t i = 0; i < S; i++){
    if(frequency[i] == 0) continue;
    frequency[i] = static_cast<double>(frequency[i]) / static_cast<double>(total);
    entropy -= frequency[i] * std::log(frequency[i]);
  };

  return {entropy, frequency};
};



void LLM::LLM::setModel() noexcept {
  model.setLearningRate(learning_rate);
  model.setActivation<0, NN::ReLU>();
  model.setActivation<1, NN::ReLU>();
  model.setActivation<2, NN::Softmax>();
  model.setLoss<2, NN::CrossEntropy>();
};



void LLM::LLM::learningInfo(bool show_learning_ifno, const std::array<double, alphabet_size>& target, unsigned char expected, double &ce_sum, double &mse_sum, size_t &correct, size_t &log_samples) noexcept {
  double* result = model.getResult();
  size_t predicted = 0;
  for(size_t k = 1; k < alphabet_size; k++)
    if(result[k] > result[predicted])
      predicted = k;

  double mse = 0.0;
  for(size_t k = 0; k < alphabet_size; k++){
    double diff = result[k] - target[k];
    mse += diff * diff / 2.0;
  };

  mse /= alphabet_size;

  double probability = result[expected];
  double ce = -std::log(std::max(probability, 1e-12));

  mse_sum += mse;
  ce_sum += ce;
  correct += predicted == expected;
  log_samples++;

  if(show_learning_ifno){
    fmt::println("");
    fmt::println(
      "Average MSE: {}, CE: {}, Accuracy: {}%",
      mse_sum / log_samples,
      ce_sum / log_samples,
      100.0 * correct / log_samples
    );

    mse_sum = 0.0;
    ce_sum = 0.0;

    correct = 0;
    log_samples = 0;
  };
};



// =========================== //
// ======= Application ======= //
// =========================== //
void LLM::LLM::learn() noexcept {
  if(!std::filesystem::exists(path_to_learning_set) || std::filesystem::is_directory(path_to_learning_set)) return;

  fmt::println(fg(fmt::color::yellow), "-- Prepering Data");
  std::string text_io = readLearningSet();
  std::string text = prepareLearningSet(text_io);
  if(text.size() <= context_size) return;

  std::pair<double, std::array<double, alphabet_size>> entropy = calculateLearningSetEntropy<alphabet_size>(text);

  if(additional_acuracy_show){
    fmt::println(fg(fmt::color::yellow), "-- Entropy");
    fmt::println(fg(fmt::color::white), "Dataset entropy: {}", entropy.first);
    for(unsigned int i = 0; i < alphabet_size; i++) fmt::println("{} - {}%", (char)(i), entropy.second[i] * 100);
  };

  fmt::println(fg(fmt::color::yellow), "-- Learning");
  fmt::println(fg(fmt::color::white), "enter to abort without lossing");
  setModel();

  std::array<double, input_size> input;
  std::array<double, alphabet_size> target;
  std::array<std::vector<size_t>, alphabet_size> character_positions;
  std::vector<unsigned char> available_characters;
  std::vector<size_t> learning_set(learn_samples);

  for(size_t i = context_size; i < text.size(); i++){
    unsigned char character = static_cast<unsigned char>(text[i]);
    if(character >= alphabet_size) character = '?';
    character_positions[character].push_back(i);
  };

  for(unsigned int i = 0; i < alphabet_size; i++)
    if(!character_positions[i].empty()) available_characters.push_back(i);

  if(available_characters.empty()) return;

  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++){
      unsigned char character = available_characters[rand() % available_characters.size()];
      const std::vector<size_t>& positions = character_positions[character];
      learning_set[i] = positions[rand() % positions.size()];
    };

    fmt::println("");
    fmt::println(fg(fmt::color::yellow), "-- Epoch {}/{}", j + 1, epoch);

    for(unsigned int r = 0; r < learning_set_repeats; r++){
      double ce_sum = 0.0;
      double mse_sum = 0.0;
      size_t correct = 0;
      size_t log_samples = 0;

      fmt::println(fg(fmt::color::white), "-- Repeat {}/{}", r + 1, learning_set_repeats);

      for(unsigned int i = 0; i < learn_samples; i++) {
        if(hasInput()){
          std::string input_io;
          std::getline(std::cin, input_io);
          fmt::println("");
          fmt::println(fg(fmt::color::yellow), "-- Learning aborted");
          saveModelToFile();
          return;
        };

        input.fill(0.0);
        target.fill(0.0);

        size_t expected_position = learning_set[i];
        size_t offset = expected_position - context_size;

        for(size_t k = 0; k < context_size; k++){
          unsigned char character = static_cast<unsigned char>(text[offset + k]);
          if(character >= alphabet_size) character = '?';
          input[k * alphabet_size + character] = 1.0;
        };

        unsigned char expected = static_cast<unsigned char>(text[expected_position]);
        if(expected >= alphabet_size) expected = '?';
        target[expected] = 1.0;

        model.setInput(input);
        model.forward();

        if(additional_acuracy_show){
          bool show_learning_ifno = ((i + 1) % 100 == 0 || i + 1 == learn_samples);
          learningInfo(show_learning_ifno, target, expected, ce_sum, mse_sum, correct, log_samples);
        };

        model.backprop(target);
        NN::Utils::progressBar(i + r * learn_samples + 1, learn_samples * learning_set_repeats);
      };
    };

    saveModelToFile();
  };

  saveModelToFile();
};



std::string LLM::LLM::getRespond(const std::string &message) noexcept {
  std::string context = message;
  std::string response = "";

  setModel();

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
  fmt::println(fg(fmt::color::blue), "set_learn_rate - to set learn rate");
  fmt::println(fg(fmt::color::blue), "set_learn_epoch - to set learn epoch");
  fmt::println(fg(fmt::color::blue), "set_learn_samples - to set learn samples");
  fmt::println(fg(fmt::color::blue), "learning_set_repeats - to set learn repeats");
  fmt::println(fg(fmt::color::blue), "set_learn_additional_log - to set learn logs");
  fmt::println(fg(fmt::color::blue), "print_model - print model");
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

  if(input == "set_learn_rate"){
    std::string new_rate = "";
    fmt::print(fg(fmt::color::white), "new_rate > ");
    std::getline(std::cin, new_rate);
    
    double new_rate_val = 0;
    try{
      new_rate_val = std::stod(new_rate);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid double");
      return;
    };
    
    setLearningRate(new_rate_val);
    return;
  };

  if(input == "set_learn_epoch"){
    std::string new_epoch = "";
    fmt::print(fg(fmt::color::white), "new_epoch > ");
    std::getline(std::cin, new_epoch);
    
    int new_epoch_val = 0;
    try{
      new_epoch_val = std::stoi(new_epoch);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setLearningEpoch(new_epoch_val);
    return;
  };

  if(input == "set_learn_samples"){
    std::string new_samples = "";
    fmt::print(fg(fmt::color::white), "new_samples > ");
    std::getline(std::cin, new_samples);
    
    int new_samples_val = 0;
    try{
      new_samples_val = std::stoi(new_samples);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setLearningSamples(new_samples_val);
    return;
  };

  if(input == "learning_set_repeats"){
    std::string new_repeat = "";
    fmt::print(fg(fmt::color::white), "new_repeat > ");
    std::getline(std::cin, new_repeat);
    
    int new_repeat_val = 0;
    try{
      new_repeat_val = std::stoi(new_repeat);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    setLearningSetRepeats(new_repeat_val);
    return;
  };

  if(input == "set_learn_additional_log"){
    std::string new_samples = "";
    fmt::print(fg(fmt::color::white), "new_logs > ");
    std::getline(std::cin, new_samples);
    
    int new_samples_val = 0;
    try{
      new_samples_val = std::stoi(new_samples);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setAdditionalAcuracyShow(new_samples_val);
    return;
  };

  if(input == "print_model"){
    fmt::println(fg(fmt::color::blue), "{}", model.print());
    return;
  };

  std::string res = getRespond(input);
  fmt::println(fg(fmt::color::azure), "{}", res);
};



void LLM::LLM::onEnd() noexcept {
  saveModelToFile();
};



void LLM::LLM::application() noexcept {
  onStart();
  while(is_running) onUpdate();
  onEnd();
};
