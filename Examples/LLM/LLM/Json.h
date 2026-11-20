#pragma once
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

struct Assertion {
  std::string type;
  std::string value;
};

struct Vars {
  std::string question;
  std::string answer;
};

struct EvalItem {
  Vars vars;
  std::vector<Assertion> assert;
};

inline void from_json(const json& j, Assertion& a) {
  j.at("type").get_to(a.type);
  j.at("value").get_to(a.value);
}

inline void from_json(const json& j, Vars& v) {
  v.question = j.value("question", "");
  v.answer = j.value("answer", "");
};

inline void from_json(const json& j, EvalItem& item) {
  j.at("vars").get_to(item.vars);
  if (j.contains("assert") && j["assert"].is_array()) j.at("assert").get_to(item.assert);
};

