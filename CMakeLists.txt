cmake_minimum_required(VERSION 3.10)
project(WarfareTacticsModel)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

include_directories(include)

add_executable(pallas
    src/main.cpp
    src/data_pipeline.cpp
    src/dataloader.cpp
    src/train_config.cpp
    src/tensor.cpp
    src/linear.cpp
    src/model.cpp
    src/logging.cpp
    src/simulation_engine.cpp
)

target_link_libraries(pallas OpenMP::OpenMP_CXX Threads::Threads dl)
target_link_libraries(pallas nlohmann_json::nlohmann_json)

add_executable(pallas_battle
    src/battle_main.cpp
    src/battle_runtime.cpp
    src/battle_server.cpp
    src/scenario_config.cpp
    src/tournament.cpp
    src/simulation_engine.cpp
    src/model.cpp
    src/linear.cpp
    src/tensor.cpp
)
target_link_libraries(pallas_battle OpenMP::OpenMP_CXX Threads::Threads)
target_link_libraries(pallas_battle nlohmann_json::nlohmann_json)

add_executable(pallas_replay
    src/replay_main.cpp
    src/battle_runtime.cpp
    src/scenario_config.cpp
    src/simulation_engine.cpp
    src/model.cpp
    src/linear.cpp
    src/tensor.cpp
)
target_link_libraries(pallas_replay OpenMP::OpenMP_CXX Threads::Threads)
target_link_libraries(pallas_replay nlohmann_json::nlohmann_json)

add_library(default_tools_plugin SHARED
    plugins/default_tools.cpp
    src/logging.cpp
)
set_target_properties(default_tools_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "default_tools"
)

add_executable(pallas_tests
    tests/model_tests.cpp
    src/train_config.cpp
    src/tensor.cpp
    src/linear.cpp
    src/model.cpp
    src/simulation_engine.cpp
    src/battle_runtime.cpp
)
target_link_libraries(pallas_tests OpenMP::OpenMP_CXX Threads::Threads)
target_link_libraries(pallas_tests nlohmann_json::nlohmann_json)
add_test(NAME pallas_unit_tests COMMAND pallas_tests)