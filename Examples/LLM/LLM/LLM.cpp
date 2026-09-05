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
    model.setWeights(-0.05, 0.05);
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

  text = std::move(normalized_text);

  std::array<size_t, alphabet_size> frequency{};
  size_t total = 0;

  for(unsigned char c : text){
    if(c >= alphabet_size) c = '?';
    frequency[c]++;
    total++;
  };

  double entropy = 0.0;

  for(size_t i = 0; i < alphabet_size; i++){
    if(frequency[i] == 0) continue;

    double p =
      static_cast<double>(frequency[i]) /
      static_cast<double>(total);

    entropy -= p * std::log(p);
  };

  fmt::println(
    "Dataset entropy: {}, Space frequency: {}%",
    entropy,
    100.0 * static_cast<double>(frequency[' ']) / total
  );

  if(text.size() <= context_size) return;

  fmt::println(fg(fmt::color::yellow), "-- Learning");
  fmt::println(fg(fmt::color::white), "enter to abort without lossing");

  std::array<double, input_size> input;
  std::array<double, alphabet_size> target;

  model.setLearningRate(learning_rate);

  model.setActivation<1, NN::Sigmoid>();
  model.setActivation<2, NN::Sigmoid>();
  model.setActivation<3, NN::Linear>();
  model.setLoss<3, NN::MSE>();

  double ce_sum = 0.0;
  size_t correct = 0;
  size_t log_samples = 0;

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

      model.setInput(input);
      model.forward();

      if(additional_acuracy_show){
        double* logits = model.getResult();

        double max_logit = logits[0];
        size_t predicted = 0;

        for(size_t k = 1; k < alphabet_size; k++){
          if(logits[k] > max_logit)
            max_logit = logits[k];

          if(logits[k] > logits[predicted])
            predicted = k;
        };

        double sum = 0.0;

        for(size_t k = 0; k < alphabet_size; k++)
          sum += std::exp(logits[k] - max_logit);

        double probability =
          std::exp(logits[expected] - max_logit) / sum;

        ce_sum += -std::log(std::max(probability, 1e-12));
        correct += predicted == expected;
        log_samples++;

        if((j * learn_samples + i + 1) % 100 == 0){
          fmt::println("");
          fmt::println(
            "Average CE: {}, Accuracy: {}%",
            ce_sum / log_samples,
            100.0 * correct / log_samples
          );

          ce_sum = 0.0;
          correct = 0;
          log_samples = 0;
        };
      }
      
      model.backprop(target);
      
      NN::Utils::progressBar(i + j * learn_samples, learn_samples * epoch);
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
  fmt::println(fg(fmt::color::blue), "set_learn_rate - to set learn rate");
  fmt::println(fg(fmt::color::blue), "set_learn_epoch - to set learn epoch");
  fmt::println(fg(fmt::color::blue), "set_learn_samples - to set learn samples");
  fmt::println(fg(fmt::color::blue), "set_learn_additional_log - to set learn logs");
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
  }
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
