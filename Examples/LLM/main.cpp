// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <memory>
#include "LLM/LLM.h"



int main(){
  std::unique_ptr<LLM::LLM> llm = std::make_unique<LLM::LLM>();
  llm->application();
  return 0;
};
