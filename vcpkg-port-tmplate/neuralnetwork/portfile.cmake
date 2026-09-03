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
    -DNEURAL_NETOWRK_BUILD_EXAMPLE=OFF
    -DNEURAL_NETOWRK_BUILD_TESTS=OFF
)

vcpkg_cmake_install()
