#include "common_utils.h"

namespace pallas {
namespace util {

bool contains_id(const std::vector<uint16_t>& values, uint16_t id) {
    return std::find(values.begin(), values.end(), id) != values.end();
}

void add_unique_id(std::vector<uint16_t>* values, uint16_t id) {
    if (values == nullptr || id == 0 || contains_id(*values, id)) {
        return;
    }
    values->push_back(id);
}

void erase_id(std::vector<uint16_t>* values, uint16_t id) {
    if (values == nullptr) {
        return;
    }
    values->erase(std::remove(values->begin(), values->end(), id), values->end());
}

std::string trim_copy(const std::string& text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string to_lower_ascii(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += "?";
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

}  // namespace util
}  // namespace pallas
