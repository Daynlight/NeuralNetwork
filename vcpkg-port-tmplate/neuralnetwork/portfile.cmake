# Neural Network
# Copyright 2026 Daynlight
# Licensed under the GNU General.
# See LICENSE file for details.



vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO Daynlight/NeuralNetwork

  REF 
  SHA512 "0"
  HEAD_REF dev
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DNEURAL_NETWORK_BUILD_EXAMPLES=OFF
    -DNEURAL_NETWORK_BUILD_TESTS=OFF
)

vcpkg_check_features()
vcpkg_cmake_install()
