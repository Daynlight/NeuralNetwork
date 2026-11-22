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
LLM::LLM::LLM() noexcept {
  embedding.zero(pad_token);
};



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



void LLM::LLM::setLearningRate(float value) noexcept {
  if(value < 0.00000001) return;
  learning_rate = value;
  model.setLearningRate(learning_rate);
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



void LLM::LLM::setBalancedLearning(int value) noexcept {
  if(value <= 0) balanced_learning = false;
  else balanced_learning = true;
};



void LLM::LLM::setGpuAcceleration(int value) noexcept {
  if(value <= 0) gpu_acceleration = false;
  else gpu_acceleration = true;
};



void LLM::LLM::setDynamicLearningRate(int value) noexcept {
  if(value <= 0) dynamic_lr = false;
  else dynamic_lr = true;
};



void LLM::LLM::setAdditionalAcuracyShow(int value) noexcept {
  if(value <= 0) additional_acuracy_show = false;
  else additional_acuracy_show = true;
};



void LLM::LLM::setMaxLearningRateReductions(int value) noexcept {
  if(value < 0) return;
  max_lr_reductions = value;
};



void LLM::LLM::setModelFilePath(std::filesystem::path path) noexcept {
  if(!std::filesystem::exists(std::filesystem::path(__FILE__).parent_path() / path) || 
      std::filesystem::is_directory(std::filesystem::path(__FILE__).parent_path() / path)) return;
  path_to_model_data = std::filesystem::path(__FILE__).parent_path() / path;
};



std::filesystem::path LLM::LLM::getModelFilePath() const noexcept {
  return path_to_model_data;
};



void LLM::LLM::updateInfo() noexcept {
  if(!std::filesystem::exists(path_to_info) || std::filesystem::is_directory(path_to_info)) return;
  std::error_code ec;
  auto last_write = std::filesystem::last_write_time(path_to_info, ec);
  
  if(ec || last_write == info_last_write) return;
  
  info_last_write = last_write;
  loadInfoFromFile();
};



// =========================== //
// ====== Serialization ====== //
// =========================== //
void LLM::LLM::loadModelFromFile() noexcept {
  if(!std::filesystem::exists(path_to_model_data) || std::filesystem::is_directory(path_to_model_data)){
    // model.setWeights<0>(-0.00968, 0.00968);
    // model.setWeights<1>(-0.02706, 0.02706);
    // model.setWeights<2>(-0.02706, 0.02706);
    // model.setWeights<3>(-0.06275, 0.06275);
    // model.setWeights<4>(-0.06275, 0.06275);
    // model.setWeights<5>(-0.03827, 0.03827);
    // model.setWeights<6>(-0.02706, 0.02706);
    // model.setWeights<7>(-0.06275, 0.06275);
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
  if(!loadEmbeddingFromFile()){
    fmt::println(fg(fmt::color::red), "Embedding missing or invalid for existing model");
    return;
  };
  loadInfoFromFile();
};



void LLM::LLM::saveModelToFile() noexcept {
  updateInfo();
  std::ofstream file(path_to_model_data, std::ios::binary);
  if(!file.is_open()) return;
  std::string data = model.serialize();

  if(!data.empty()) file.write(data.data(), data.size());
  saveEmbeddingToFile();
  saveInfoToFile();
};



void LLM::LLM::saveEmbeddingToFile() const noexcept {
  std::filesystem::path path = path_to_embedding;

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if(!file.is_open()) return;

  std::string data = embedding.serialize();
  if(!data.empty()) file.write(data.data(), data.size());
};



bool LLM::LLM::loadEmbeddingFromFile() noexcept {
  std::filesystem::path path = path_to_embedding;

  if(!std::filesystem::exists(path) || std::filesystem::is_directory(path)) return false;

  std::ifstream file(path, std::ios::binary);
  if(!file.is_open()) return false;

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  if(size != embedding.size() * sizeof(float)) return false;

  std::string data(size, '\0');

  if(size > 0)
    file.read(data.data(), size);

  if(!file) return false;
  if(!embedding.deserialize(data)) return false;

  embedding.zero(pad_token);

  return true;
};



void LLM::LLM::saveInfoToFile() noexcept {
  std::ofstream file(path_to_info);

  if(!file) return;

  size_t model_nodes = 0;
  size_t model_parameters = 0;

  for(size_t i = 1; i < model_shape.size(); i++)
    model_nodes += model_shape[i];

  for(size_t i = 0; i < model_shape.size() - 1; i++)
    model_parameters += (model_shape[i] + 1) * model_shape[i + 1];

  constexpr size_t embedding_parameters = vocab_size * embedding_size;

  size_t total_parameters = model_parameters + embedding_parameters;

  file << "[training]\n";
  file << "learning_rate=" << learning_rate << '\n';
  file << "epoch=" << epoch << '\n';
  file << "learn_samples=" << learn_samples << '\n';
  file << "learning_set_repeats=" << learning_set_repeats << '\n';
  file << "max_lr_reductions=" << max_lr_reductions << '\n';
  file << "balanced_learning=" << balanced_learning << '\n';
  file << "dynamic_lr=" << dynamic_lr << '\n';
  file << "gpu_acceleration=" << gpu_acceleration << '\n';
  file << "additional_accuracy_show=" << additional_acuracy_show << '\n';

  file << '\n';

  file << "[statistics]\n";
  file << "update_counter=" << update_counter << '\n';
  file << "request_counter=" << request_counter << '\n';
  file << "last_ce=" << last_ce << '\n';
  file << "best_ce=" << best_ce << '\n';
  file << "last_mse=" << last_mse << '\n';
  file << "best_mse=" << best_mse << '\n';
  file << "last_accuracy=" << last_accuracy << '\n';
  file << "best_accuracy=" << best_accuracy << '\n';
  file << "last_update_time_ms=" << last_update_time_ms << '\n';
  file << "total_update_time_ms=" << total_update_time_ms << '\n';
  file << "average_update_time_ms=" << (update_counter ? total_update_time_ms / update_counter : 0.0) << '\n';

  file << '\n';

  file << "[architecture]\n";
  file << "context_size=" << context_size << '\n';
  file << "embedding_size=" << embedding_size << '\n';
  file << "input_size=" << input_size << '\n';
  file << "vocab_size=" << vocab_size << '\n';
  file << "response_size=" << response_size << '\n';

  file << '\n';

  file << "model_layers=" << model_shape.size() - 1 << '\n';
  file << "model_nodes=" << model_nodes << '\n';
  file << "model_parameters=" << model_parameters << '\n';
  file << "embedding_parameters=" << embedding_parameters << '\n';
  file << "total_parameters=" << total_parameters << '\n';

  file << "model_size_bytes=" << model_parameters * sizeof(float) << '\n';
  file << "embedding_size_bytes=" << embedding_parameters * sizeof(float) << '\n';
  file << "total_parameter_size_bytes=" << total_parameters * sizeof(float) << '\n';

  file << "model_shape=";

  for(size_t i = 0; i < model_shape.size(); i++){
    if(i) file << ',';
    file << model_shape[i];
  };

  file << '\n';

  file.close();

  std::error_code ec;
  auto last_write = std::filesystem::last_write_time(path_to_info, ec);

  if(!ec) info_last_write = last_write;
};



void LLM::LLM::loadInfoFromFile() noexcept {
  if(!std::filesystem::exists(path_to_info) || std::filesystem::is_directory(path_to_info)) return;

  std::ifstream file(path_to_info);

  if(!file) return;

  std::string line;

  while(std::getline(file, line)){

    if(line.empty() || line.front() == '[') continue;

    size_t pos = line.find('=');

    if(pos == std::string::npos) continue;

    std::string_view key(line.data(), pos);
    std::string_view value(line.data() + pos + 1, line.size() - pos - 1);

    if(key == "learning_rate"){

      float data = 0.0f;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setLearningRate(data);

    }
    else if(key == "epoch"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setLearningEpoch(data);

    }
    else if(key == "learn_samples"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setLearningSamples(data);

    }
    else if(key == "learning_set_repeats"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setLearningSetRepeats(data);

    }
    else if(key == "max_lr_reductions"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setMaxLearningRateReductions(data);

    }
    else if(key == "balanced_learning"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setBalancedLearning(data);

    }
    else if(key == "dynamic_lr"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setDynamicLearningRate(data);

    }
    else if(key == "gpu_acceleration"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setGpuAcceleration(data);

    }
    else if(key == "additional_accuracy_show"){

      int data = 0;

      if(std::from_chars(value.data(), value.data() + value.size(), data).ec == std::errc())
        setAdditionalAcuracyShow(data);

    }
    else if(key == "update_counter")
      std::from_chars(value.data(), value.data() + value.size(), update_counter);

    else if(key == "request_counter")
      std::from_chars(value.data(), value.data() + value.size(), request_counter);

    else if(key == "last_ce")
      std::from_chars(value.data(), value.data() + value.size(), last_ce);

    else if(key == "best_ce")
      std::from_chars(value.data(), value.data() + value.size(), best_ce);

    else if(key == "last_mse")
      std::from_chars(value.data(), value.data() + value.size(), last_mse);

    else if(key == "best_mse")
      std::from_chars(value.data(), value.data() + value.size(), best_mse);

    else if(key == "last_accuracy")
      std::from_chars(value.data(), value.data() + value.size(), last_accuracy);

    else if(key == "best_accuracy")
      std::from_chars(value.data(), value.data() + value.size(), best_accuracy);

    else if(key == "last_update_time_ms")
      std::from_chars(value.data(), value.data() + value.size(), last_update_time_ms);

    else if(key == "total_update_time_ms")
      std::from_chars(value.data(), value.data() + value.size(), total_update_time_ms);
  };

  file.close();

  std::error_code ec;
  auto last_write = std::filesystem::last_write_time(path_to_info, ec);

  if(!ec) info_last_write = last_write;
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
  
   json parsed;
  try{
    parsed = json::parse(text);
    json_items = parsed.get<std::vector<EvalItem>>();
  }
  catch(...){
    fmt::println(fg(fmt::color::red), "Invalid JSON learning set");
    json_items.clear();
    return "";
  };

  return text;
};




std::string LLM::LLM::prepareLearningSet(const std::string& text) const noexcept {
  std::stringstream stream(text);

  std::string line;
  std::string compact_text;

  while(std::getline(stream, line)){
    while(!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
      line.pop_back();
    if(line.empty()) continue;
    if(line.back() == ':') continue;
    if(!compact_text.empty()) compact_text += ' ';
    compact_text += line;
  };

  auto isPunct = [](unsigned char c){
    return c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':' ||
           c == '(' || c == ')' || c == '"';
  };

  std::string normalized_text;
  normalized_text.reserve(compact_text.size() * 2);
  bool last_was_space = true;

  for(size_t i = 0; i < compact_text.size(); i++){
    unsigned char c = static_cast<unsigned char>(compact_text[i]);

    if(std::isspace(c)){
      if(!last_was_space) normalized_text += ' ';
      last_was_space = true;
      continue;
    };

    if(isPunct(c)){
      if(!last_was_space) normalized_text += ' ';
      normalized_text += static_cast<char>(c);
      normalized_text += ' ';
      last_was_space = true;
      continue;
    };

    normalized_text += (c < 128) ? static_cast<char>(std::tolower(c)) : static_cast<char>(c);
    last_was_space = false;
  };

  while(!normalized_text.empty() && normalized_text.back() == ' ')
    normalized_text.pop_back();

  return normalized_text;
};




template<size_t S>
std::pair<float, std::array<float, S>> LLM::LLM::calculateLearningSetEntropy(const std::string &text) const noexcept {
  std::array<float, S> frequency{};
  size_t total = 0;

  for(unsigned char c : text){
    if(c >= S) c = '?';
    frequency[c]++;
    total++;
  };

  float entropy = 0.0;
  for(size_t i = 0; i < S; i++){
    if(frequency[i] == 0) continue;
    frequency[i] = static_cast<float>(frequency[i]) / static_cast<float>(total);
    entropy -= frequency[i] * std::log(frequency[i]);
  };

  return {entropy, frequency};
};



void LLM::LLM::setModel() noexcept {
  model.setLearningRate(learning_rate);
  model.setActivation<0, NN::Linear>();
  model.setActivation<1, NN::ReLU>();
  model.setActivation<2, NN::ReLU>();
  model.setActivation<3, NN::ReLU>();
  model.setActivation<4, NN::ReLU>();
  model.setActivation<5, NN::Softmax>();
  model.setLoss<5, NN::CrossEntropy>();
  model.setGPUAcceleration(gpu_acceleration);
};



void LLM::LLM::learningInfo(bool show_learning_ifno, const std::array<float, vocab_size>& target, size_t expected, float &ce_sum, float &mse_sum, size_t &correct, size_t &log_samples) noexcept {
  float* result = model.getActivatedResult();

  size_t predicted = 0;

  for(size_t k = 1; k < vocab_size; k++) if(result[k] > result[predicted]) predicted = k;
  
  float mse = 0.0;
  for(size_t k = 0; k < vocab_size; k++){
    float diff = result[k] - target[k];
    mse += diff * diff / 2.0;
  };
  mse /= vocab_size;

  float probability = result[expected];
  float ce = -std::log(std::max(probability, 0.0000001f));

  mse_sum += mse;
  ce_sum += ce;
  correct += predicted == expected;
  log_samples++;

  if(show_learning_ifno){
    float average_mse = mse_sum / log_samples;
    float average_ce = ce_sum / log_samples;
    float accuracy = 100.0f * correct / log_samples;

    fmt::println("");
    fmt::println(
      "Average MSE: {}, CE: {}, Accuracy: {}%",
      mse_sum / log_samples,
      ce_sum / log_samples,
      100.0 * correct / log_samples
    );

      last_ce = average_ce;
    last_mse = average_mse;
    last_accuracy = accuracy;

    best_ce = std::min(best_ce, average_ce);
    best_mse = std::min(best_mse, average_mse);
    best_accuracy = std::max(best_accuracy, accuracy);

    mse_sum = 0.0;
    ce_sum = 0.0;
    correct = 0;
    log_samples = 0;
  };
};



void LLM::LLM::createTokens(const std::string& text) noexcept {
  if(tokens.empty()){
    tokens.emplace_back("<UNK>");
    tokens.emplace_back("<PAD>");
    tokens.emplace_back("<EOS>");
    tokens.emplace_back("<SEP>");
    token_to_id.emplace("<UNK>", unk_token);
    token_to_id.emplace("<PAD>", pad_token);
    token_to_id.emplace("<EOS>", eos_token);
    token_to_id.emplace("<SEP>", sep_token);
  }
  else if(tokens.size() <= sep_token || tokens[eos_token] != "<EOS>" || tokens[sep_token] != "<SEP>"){
    fmt::println(fg(fmt::color::red), "Vocabulary has no <EOS>/<SEP> at ids {}/{} - delete the old tokens and model files", eos_token, sep_token);
    return;
  }

  std::unordered_map<std::string, size_t> frequency;
  std::istringstream stream(text);
  std::string word;

  while(stream >> word){
    if(token_to_id.contains(word)) continue;
    frequency[word]++;
  }

  std::vector<std::pair<std::string, size_t>> sorted;
  sorted.reserve(frequency.size());

  for(const auto& [word, count] : frequency) sorted.emplace_back(word, count);
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){
    if(a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });

  size_t next = sep_token + 1;

  for(const auto& [word, count] : sorted){
    while(next < tokens.size() && tokens[next].rfind("<UNUSED_", 0) != 0) next++;
    if(next >= vocab_size) break;

    if(next < tokens.size()){
      token_to_id.erase(tokens[next]);
      tokens[next] = word;
    }
    else tokens.push_back(word);

    token_to_id.emplace(word, next);
    next++;
  }

  while(tokens.size() < vocab_size){
    std::string token = "<UNUSED_" + std::to_string(tokens.size()) + ">";
    token_to_id.emplace(token, tokens.size());
    tokens.push_back(std::move(token));
  }
};



void LLM::LLM::saveTokensToFile() const noexcept {
  std::ofstream file(path_to_tokens, std::ios::binary | std::ios::trunc);
  if(!file.is_open()) return;

  uint64_t count = tokens.size();
  file.write(reinterpret_cast<const char*>(&count), sizeof(count));

  for(const std::string& token : tokens){
    uint64_t size = token.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(token.data(), size);
  };
};



bool LLM::LLM::loadTokensFromFile() noexcept {
  std::ifstream file(path_to_tokens, std::ios::binary);
  if(!file.is_open()) return false;

  uint64_t count = 0;
  file.read(reinterpret_cast<char*>(&count), sizeof(count));
  if(!file || count != vocab_size) return false;

  std::vector<std::string> loaded_tokens;
  std::unordered_map<std::string, size_t> loaded_ids;

  for(uint64_t i = 0; i < count; i++){
    uint64_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    if(!file || size > 4096) return false;

    std::string token(size, '\0');

    if(size > 0) file.read(token.data(), size);
    if(!file) return false;
    if(loaded_ids.contains(token)) return false;

    loaded_ids.emplace(token, loaded_tokens.size());
    loaded_tokens.push_back(std::move(token));
  };

  if(loaded_tokens[unk_token] != "<UNK>" || loaded_tokens[pad_token] != "<PAD>" || loaded_tokens[eos_token] != "<EOS>" || loaded_tokens[sep_token] != "<SEP>") return false;

  tokens = std::move(loaded_tokens);
  token_to_id = std::move(loaded_ids);

  return true;
};



std::vector<size_t> LLM::LLM::encodeTokens(const std::string& text) const noexcept {
  std::vector<size_t> result;
  std::istringstream stream(text);
  std::string word;

  while(stream >> word){
    auto it = token_to_id.find(word);

    if(it == token_to_id.end()) result.push_back(unk_token);
    else result.push_back(it->second);
  };

  return result;
};



std::string LLM::LLM::decodeTokens(const std::vector<size_t>& ids) const noexcept {
  std::string result;

  for(size_t id : ids){
    if(id >= tokens.size()) continue;
    if(id == pad_token || id == unk_token || id == eos_token || id == sep_token) continue;

    const std::string& token = tokens[id];
    if(token.rfind("<UNUSED_", 0) == 0) continue;

    const bool closing = token == "." || token == "," || token == "!" ||
                         token == "?" || token == ";" || token == ":" || token == ")";

    if(!result.empty() && !closing && result.back() != '(') result += ' ';
    result += token;
  };

  return result;
};



// =========================== //
// ======= Application ======= //
// =========================== //
void LLM::LLM::learn() noexcept {
  if(!std::filesystem::exists(path_to_learning_set) || std::filesystem::is_directory(path_to_learning_set)) return;

  auto printParameters = [&]{
    fmt::println(fg(fmt::color::yellow), "-- Parameters");
    fmt::println(fg(fmt::color::blue), "Path To Learning Set: {}", path_to_learning_set);
    fmt::println(fg(fmt::color::blue), "Path To Model Data: {}", path_to_model_data);
    fmt::println(fg(fmt::color::blue), "Path To Tokens: {}", path_to_tokens);
    fmt::println(fg(fmt::color::blue), "Context Size: {}", context_size);
    fmt::println(fg(fmt::color::blue), "Vocab Size: {}", vocab_size);
    fmt::println(fg(fmt::color::blue), "Input Size: {}", input_size);
    fmt::println(fg(fmt::color::blue), "Epoch: {}", epoch);
    fmt::println(fg(fmt::color::blue), "Learn Samples: {}", learn_samples);
    fmt::println(fg(fmt::color::blue), "Learning Repeat: {}", learning_set_repeats);
    fmt::println(fg(fmt::color::blue), "Learning Rate: {}", learning_rate);
    fmt::println(fg(fmt::color::blue), "Dynamic Learning Rate: {}", dynamic_lr);
    fmt::println(fg(fmt::color::blue), "Max Dynamic Learning Rate Reduction: {}", max_lr_reductions);
    fmt::println(fg(fmt::color::blue), "Balanced Learning: {}", balanced_learning);
    fmt::println(fg(fmt::color::blue), "GPU Acceleration: {}", gpu_acceleration);
    fmt::println(fg(fmt::color::blue), "Additional Info: {}", additional_acuracy_show);
  };

  printParameters();

  fmt::println(fg(fmt::color::yellow), "-- Prepering Data");
  readLearningSet();

  std::vector<std::pair<std::string, std::string>> pairs;
  std::string token_text;

  for(const auto& item : json_items){
    if(item.vars.question.empty() || item.vars.answer.empty()) continue;

    std::string q = prepareLearningSet(item.vars.question);
    std::string a = prepareLearningSet(item.vars.answer);
    if(q.empty() || a.empty()) continue;

    token_text += q + "\n" + a + "\n";
    pairs.emplace_back(std::move(q), std::move(a));
  };

  if(pairs.empty() || context_size == 0){
    fmt::println(fg(fmt::color::red), "No valid JSON question/answer pairs found");
    return;
  };

  fmt::println(fg(fmt::color::green), "Using {} JSON question/answer pairs (of {} items)", pairs.size(), json_items.size());

  if(!loadTokensFromFile() && std::filesystem::exists(path_to_model_data)){
    fmt::println(fg(fmt::color::red), "Vocabulary missing or invalid for existing model");
    return;
  };

  createTokens(token_text);
  saveTokensToFile();
  if(tokens.size() != vocab_size) return;

  struct Sequence {
    std::vector<size_t> question;
    std::vector<size_t> answer;
  };

  struct Sample {
    size_t pair;
    size_t pos;
  };

  std::vector<Sequence> sequences;
  std::vector<Sample> valid_samples;

  for(const auto& [q, a] : pairs){
    Sequence seq;
    seq.question = encodeTokens(q);
    seq.answer = encodeTokens(a);
    if(seq.question.empty() || seq.answer.empty()) continue;

    seq.question.push_back(sep_token);
    seq.answer.push_back(eos_token);

    const size_t index = sequences.size();

    for(size_t p = 0; p < seq.answer.size(); p++){
      const size_t t = seq.answer[p];
      if(t == unk_token || t == pad_token) continue;
      valid_samples.push_back({index, p});
    };

    sequences.push_back(std::move(seq));
  };

  if(valid_samples.empty()) return;

  fmt::println(fg(fmt::color::yellow), "-- Learning");
  fmt::println(fg(fmt::color::white), "enter to abort without lossing");

  setModel();
  embedding.zero(pad_token);

  std::array<float, input_size> input{};
  std::array<float, vocab_size> target{};
  std::array<size_t, context_size> active_input{};

  size_t previous_expected = 0;
  bool has_previous_sample = false;

  float scheduler_best_ce = std::numeric_limits<float>::max();
  unsigned int stale_epochs = 0;
  unsigned int lr_reductions = 0;

  // warmup + cosine schedule (used only when dynamic_lr is off)
  const float base_lr = learning_rate;
  const size_t total_steps = static_cast<size_t>(epoch) * learning_set_repeats * learn_samples;
  const size_t warmup_steps = std::max<size_t>(1, total_steps / 20);
  size_t global_step = 0;
  float current_lr = base_lr;

  std::vector<std::vector<Sample>> token_samples(vocab_size);
  std::vector<size_t> available_tokens;
  std::vector<Sample> learning_set(learn_samples);
  std::vector<size_t> context;
  context.reserve(context_size * 4);

  for(const Sample& s : valid_samples)
    token_samples[sequences[s.pair].answer[s.pos]].push_back(s);

  for(size_t i = 0; i < vocab_size; i++)
    if(!token_samples[i].empty()) available_tokens.push_back(i);

  const size_t frequent_limit = sep_token + 1 + vocab_size / 20;
  auto chance = [](float p){ return (rand() / (float)RAND_MAX) < p; };

  auto appendQuestion = [&](const std::vector<size_t>& q){
    if(q.size() <= 3 || !chance(0.5f)){
      context.insert(context.end(), q.begin(), q.end());
      return;
    };

    const size_t before = context.size();

    for(size_t k = 0; k < q.size(); k++){
      const bool keep_always = (k + 2 >= q.size());
      const float p = (q[k] < frequent_limit) ? 0.30f : 0.08f;

      if(keep_always){
        context.push_back(q[k]);
        continue;
      };

      if(chance(p)) continue;
      context.push_back(chance(0.05f) ? unk_token : q[k]);
    };

    if(context.size() == before) context.push_back(q.back());
  };

  for(unsigned int j = 0; j < epoch; j++){
    updateInfo();
    learning_set.resize(learn_samples);
    auto last = std::chrono::steady_clock::now();

    printParameters();
    fmt::println("valid_samples: {}", valid_samples.size());

    float epoch_ce_sum = 0.0;
    size_t epoch_samples = 0;

    for(unsigned int i = 0; i < learn_samples; i++){
      if(balanced_learning){
        size_t token = available_tokens[rand() % available_tokens.size()];
        const std::vector<Sample>& samples = token_samples[token];

        learning_set[i] = samples[rand() % samples.size()];
      }
      else learning_set[i] = valid_samples[rand() % valid_samples.size()];
    };

    fmt::println("");
    fmt::println(fg(fmt::color::yellow), "-- Epoch {}/{}", j + 1, epoch);

    for(unsigned int r = 0; r < learning_set_repeats; r++){
      float ce_sum = 0.0;
      float mse_sum = 0.0;
      size_t correct = 0;
      size_t log_samples = 0;

      fmt::println(fg(fmt::color::white), "-- Repeat {}/{}", r + 1, learning_set_repeats);

      for(unsigned int i = 0; i < learn_samples; i++){
        auto start_update = std::chrono::steady_clock::now();

        if(hasInput()){
          std::string input_io;
          std::getline(std::cin, input_io);

          fmt::println("");
          fmt::println(fg(fmt::color::yellow), "-- Learning aborted");

          model.setLearningRate(learning_rate);
          saveModelToFile();
          return;
        };

        if(!dynamic_lr){
          const float progress = static_cast<float>(global_step) / static_cast<float>(std::max<size_t>(1, total_steps));
          const float warmup = std::min(1.0f, static_cast<float>(global_step + 1) / static_cast<float>(warmup_steps));
          current_lr = base_lr * warmup * (0.1f + 0.9f * 0.5f * (1.0f + std::cos(3.14159265f * progress)));
          if(global_step % 50 == 0) model.setLearningRate(current_lr);
        }
        else current_lr = learning_rate;
        global_step++;

        if(has_previous_sample)
          target[previous_expected] = 0.0;

        const Sample& sample = learning_set[i];
        const Sequence& seq = sequences[sample.pair];

        context.clear();
        appendQuestion(seq.question);
        const size_t q_len = context.size();
        context.insert(context.end(), seq.answer.begin(), seq.answer.begin() + sample.pos);
        const size_t a_len = context.size() - q_len;

        size_t q_take = std::min(q_len, context_size / 2);
        const size_t a_take = std::min(a_len, context_size - q_take);
        q_take = std::min(q_len, context_size - a_take);
        const size_t padding = context_size - q_take - a_take;

        for(size_t k = 0; k < padding; k++)
          active_input[k] = pad_token;

        for(size_t k = 0; k < q_take; k++)
          active_input[padding + k] = context[q_len - q_take + k];

        for(size_t k = 0; k < a_take; k++)
          active_input[padding + q_take + k] = context[context.size() - a_take + k];

        embedding.forward(active_input, input);

        const size_t expected = seq.answer[sample.pos];

        target[expected] = 1.0;
        previous_expected = expected;
        has_previous_sample = true;

        model.setInput(input);
        model.forward();

        if(dynamic_lr){
          float* result = model.getActivatedResult();

          epoch_ce_sum += -std::log(std::max(result[expected], 0.0000001f));
          epoch_samples++;
        };

        if(additional_acuracy_show){
          const bool show_learning_ifno = ((i + 1) % 100 == 0 || i + 1 == learn_samples);

          learningInfo(show_learning_ifno, target, expected, ce_sum, mse_sum, correct, log_samples);
        };

        model.backprop(target);

        embedding.backprop(
          active_input,
          model.getInputSigma(),
          current_lr,
          pad_token
        );

        glFinish();
        auto end_update = std::chrono::steady_clock::now();
        update_counter++;
        last_update_time_ms = std::chrono::duration<double, std::milli>(end_update - start_update).count();
        total_update_time_ms += last_update_time_ms;

        NN::Utils::progressBar(i, learn_samples);
      };
    };

    if(dynamic_lr && epoch_samples > 0){
      float epoch_ce = epoch_ce_sum / epoch_samples;

      if(epoch_ce < scheduler_best_ce - 0.01){
        scheduler_best_ce = epoch_ce;
        stale_epochs = 0;
      }
      else stale_epochs++;

      if(stale_epochs >= 3 && lr_reductions < max_lr_reductions){
        learning_rate *= 0.5f;
        learning_rate = std::max(learning_rate, 0.00001f);

        model.setLearningRate(learning_rate);

        scheduler_best_ce = epoch_ce;
        stale_epochs = 0;
        lr_reductions++;

        fmt::println(fg(fmt::color::purple), "-- Learning rate reduced to {}", learning_rate);
      };
    };

    glFinish();

    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - last).count();
    fmt::println("Elapsed: {:.3f}s", elapsed);

    saveModelToFile();
  };

  model.setLearningRate(learning_rate);
  saveModelToFile();
};



std::string LLM::LLM::getRespond(const std::string &message) noexcept {
  if(tokens.size() != vocab_size) return "";

  std::vector<size_t> question = encodeTokens(prepareLearningSet(message));
  if(question.empty()) return "";
  question.push_back(sep_token);

  constexpr float temperature = 0.6f;
  constexpr float top_p = 0.9f;
  constexpr size_t top_k = 20;
  constexpr float repetition_penalty = 1.1f;
  constexpr size_t repetition_window = 8;

  static thread_local std::mt19937 rng{std::random_device{}()};

  auto banned = [&](size_t id){
    return id == unk_token || id == pad_token || id == sep_token ||
           tokens[id].rfind("<UNUSED_", 0) == 0;
  };

  std::vector<size_t> response;
  response.reserve(response_size);

  setModel();

  std::array<float, input_size> input{};
  std::array<size_t, context_size> active_input{};
  std::vector<std::pair<float, size_t>> candidates;
  candidates.reserve(vocab_size);

  for(size_t i = 0; i < response_size; i++){
    const size_t q_len = question.size();
    const size_t a_len = response.size();

    size_t q_take = std::min(q_len, context_size / 2);
    const size_t a_take = std::min(a_len, context_size - q_take);
    q_take = std::min(q_len, context_size - a_take);
    const size_t padding = context_size - q_take - a_take;

    for(size_t k = 0; k < padding; k++)
      active_input[k] = pad_token;

    for(size_t k = 0; k < q_take; k++)
      active_input[padding + k] = question[q_len - q_take + k];

    for(size_t k = 0; k < a_take; k++)
      active_input[padding + q_take + k] = response[a_len - a_take + k];

    embedding.forward(active_input, input);

    model.setInput(input);
    model.forward();

    const float* result = model.getActivatedResult();

    candidates.clear();

    const size_t recent_begin = response.size() > repetition_window
      ? response.size() - repetition_window
      : 0;

    for(size_t j = 0; j < vocab_size; j++){
      if(banned(j)) continue;
      if(j == eos_token && response.empty()) continue;

      float p = result[j];

      if(std::find(response.begin() + recent_begin, response.end(), j) != response.end())
        p /= repetition_penalty;

      candidates.emplace_back(std::pow(std::max(p, 1e-9f), 1.0f / temperature), j);
    };

    if(candidates.empty()) break;

    const size_t k = std::min(top_k, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
      [](const auto& a, const auto& b){ return a.first > b.first; });

    float total = 0.0f;
    for(size_t c = 0; c < k; c++) total += candidates[c].first;

    size_t keep = 0;
    float cumulative = 0.0f;
    while(keep < k){
      cumulative += candidates[keep].first / total;
      keep++;
      if(cumulative >= top_p) break;
    };

    float sum = 0.0f;
    for(size_t c = 0; c < keep; c++) sum += candidates[c].first;

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float pick = dist(rng);
    size_t predicted = candidates[keep - 1].second;

    for(size_t c = 0; c < keep; c++){
      pick -= candidates[c].first;
      if(pick <= 0.0f){ predicted = candidates[c].second; break; }
    };

    if(predicted == eos_token) break;

    response.push_back(predicted);
  };

  request_counter++;

  return decodeTokens(response);
};



void LLM::LLM::printHelp() const noexcept {
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "Welcome to LLM");
  fmt::println(fg(fmt::color::blue), "help - see help message");
  fmt::println(fg(fmt::color::blue), "learn - to start learning from file");
  fmt::println(fg(fmt::color::blue), "set_learn_file - to set learn file");
  fmt::println(fg(fmt::color::blue), "set_learn_rate - to set learn rate");
  fmt::println(fg(fmt::color::blue), "set_learn_epoch - to set learn epoch");
  fmt::println(fg(fmt::color::blue), "set_learn_samples - to set learn samples");
  fmt::println(fg(fmt::color::blue), "set_learning_repeats - to set learn repeats");
  fmt::println(fg(fmt::color::blue), "set_balanced_learning - to set balanced learn");
  fmt::println(fg(fmt::color::blue), "set_gup_acceleration - to set gpu acceleration learn");
  fmt::println(fg(fmt::color::blue), "set_learn_additional_log - to set learn logs");
  fmt::println(fg(fmt::color::blue), "set_dynamic_lr - to set dynamic learning rate");
  fmt::println(fg(fmt::color::blue), "print_model - print model");
  fmt::println(fg(fmt::color::blue), "quit - exit");
};



void LLM::LLM::onStart() noexcept {
  if(!loadTokensFromFile()){
    if(std::filesystem::exists(path_to_model_data)){
      fmt::println(fg(fmt::color::red), "Vocabulary missing or invalid for existing model");
      is_running = false;
      return;
    };
  };

  std::string text = prepareLearningSet(readLearningSet());

  if(text.empty()){
    fmt::println(fg(fmt::color::red), "Learning set is empty");
    is_running = false;
    return;
  };

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
    
    float new_rate_val = 0;
    try{
      new_rate_val = std::stof(new_rate);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid float");
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

  if(input == "set_learning_repeats"){
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

  if(input == "set_dynamic_lr"){
    std::string new_dynamic_lr = "";
    fmt::print(fg(fmt::color::white), "new_dynamic_lr > ");
    std::getline(std::cin, new_dynamic_lr);
    
    int new_new_dynamic_lr_val = 0;
    try{
      new_new_dynamic_lr_val = std::stoi(new_dynamic_lr);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setDynamicLearningRate(new_new_dynamic_lr_val);
    return;
  };

  if(input == "set_balanced_learning"){
    std::string new_balanced = "";
    fmt::print(fg(fmt::color::white), "new_balanced > ");
    std::getline(std::cin, new_balanced);
    
    int new_new_balanced_val = 0;
    try{
      new_new_balanced_val = std::stoi(new_balanced);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setBalancedLearning(new_new_balanced_val);
    return;
  };

  if(input == "set_gup_acceleration"){
    std::string new_acc = "";
    fmt::print(fg(fmt::color::white), "new_acc > ");
    std::getline(std::cin, new_acc);
    
    int new_acc_val = 0;
    try{
      new_acc_val = std::stoi(new_acc);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setGpuAcceleration(new_acc_val);
    return;
  };

  if(input == "print_model"){
    fmt::println(fg(fmt::color::blue), "{}", model.print());
    return;
  };

  if(input.empty()) return;
  updateInfo();
  std::string res = getRespond(input);
  fmt::println(fg(fmt::color::azure), "{}", res);
};



void LLM::LLM::onEnd() noexcept {
  saveModelToFile();
  saveTokensToFile();
};



void LLM::LLM::application() noexcept {
  onStart();
  
  if(!is_running) return;
  while(is_running) onUpdate();
  
  onEnd();
};
