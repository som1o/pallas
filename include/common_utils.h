#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace pallas {
namespace util {

bool contains_id(const std::vector<uint16_t>& values, uint16_t id);
void add_unique_id(std::vector<uint16_t>* values, uint16_t id);
void erase_id(std::vector<uint16_t>* values, uint16_t id);

std::string trim_copy(const std::string& text);
std::string to_lower_ascii(std::string text);
std::string json_escape(const std::string& value);

template <typename T>
void write_binary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool read_binary(std::ifstream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(in);
}

}  // namespace util
}  // namespace pallas

#endif
