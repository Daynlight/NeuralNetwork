// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <iomanip>
#include <iostream>



namespace NN::Utils{
void progressBar(unsigned int current, unsigned int max_element, int barWidth = 50){
  if (current % 100 == 0 || current + 1 == max_element) {
    float progress = (float)(current) / (max_element);

    std::cout << "\r[";
    int position = barWidth * progress;

    for (int k = 0; k < barWidth; k++) {
      if (k < position + 1) std::cout << "=";
      else std::cout << " ";
    };

    std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << "%";
    std::cout.flush();
  };
  if(current + 1 == max_element) std::cout << "\n";
};
};