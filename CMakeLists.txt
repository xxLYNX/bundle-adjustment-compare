cmake_minimum_required(VERSION 3.10)
project(ba_module_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ba_module
    scripts/ba_module.cpp
)

# Eigen headers (Debian/Ubuntu layout)
target_include_directories(ba_module PRIVATE
    /usr/include/eigen3
)

# Ceres + glog headers are in /usr/include on Debian/Ubuntu
target_include_directories(ba_module PRIVATE
    /usr/include
)

# Link Ceres + glog + pthread
target_link_libraries(ba_module PRIVATE
    ceres
    glog
    pthread
)

