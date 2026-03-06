#ifndef STRATEGY_UTILS_H
#define STRATEGY_UTILS_H

#include "battle_common.h"
#include "model.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace pallas {
namespace strategy {

const std::unordered_map<std::string, Strategy>& strategy_lookup();
std::optional<Strategy> strategy_from_string(const std::string& value);

const std::unordered_map<std::string, uint32_t>& action_lookup();
uint32_t action_from_string(const std::string& value);

}  // namespace strategy
}  // namespace pallas

#endif
