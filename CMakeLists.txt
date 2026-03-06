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

add_library(pallas_core STATIC
    src/train_config.cpp
    src/tensor.cpp
    src/linear.cpp
    src/model.cpp
    src/logging.cpp
    src/simulation_engine.cpp
    src/battle_runtime.cpp
    src/scenario_config.cpp
    src/data_pipeline.cpp
    src/dataloader.cpp
    src/common_utils.cpp
    src/strategy_utils.cpp
)
target_link_libraries(pallas_core PUBLIC OpenMP::OpenMP_CXX Threads::Threads nlohmann_json::nlohmann_json)

add_executable(pallas
    src/main.cpp
)

target_link_libraries(pallas pallas_core dl)

add_executable(pallas_battle
    src/battle_main.cpp
    src/battle_server.cpp
    src/tournament.cpp
)
target_link_libraries(pallas_battle pallas_core)

add_executable(pallas_replay
    src/replay_main.cpp
)
target_link_libraries(pallas_replay pallas_core)

add_library(default_tools_plugin SHARED
    plugins/default_tools.cpp
    src/logging.cpp
    src/common_utils.cpp
)
set_target_properties(default_tools_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "default_tools"
)

add_executable(pallas_tests
    tests/model_tests.cpp
)
target_link_libraries(pallas_tests pallas_core)
add_test(NAME pallas_unit_tests COMMAND pallas_tests)