cmake_minimum_required(VERSION 3.10)
project(karate_core)

# --- FIX: Χρησιμοποιούμε το επίσημο όνομα "Python" ---
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)

# --- Setup Pybind11 ---
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# --- Create the Module ---
pybind11_add_module(karate_core karate_engine.cpp)

# --- Optimizations (Release Mode) ---
if(MSVC)
  target_compile_options(karate_core PRIVATE /O2 /fp:fast)
else()
  target_compile_options(karate_core PRIVATE -O3 -march=native)
endif()

# --- Link to Python ---
target_link_libraries(karate_core PRIVATE Python::Module)