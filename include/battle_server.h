#ifndef BATTLE_SERVER_H
#define BATTLE_SERVER_H

#include "battle_runtime.h"

#include <cstdint>
#include <string>

namespace battle {

class BattleServer {
public:
    BattleServer(BattleEngine* engine, std::string web_root, uint16_t port);

    bool run();

private:
    std::string content_type_for(const std::string& path) const;
    std::string read_text_file(const std::string& path) const;
    std::string handle_request(const std::string& method,
                               const std::string& path,
                               const std::string& body,
                               int* status,
                               std::string* content_type);

    BattleEngine* engine_;
    std::string web_root_;
    uint16_t port_;
};

}  // namespace battle

#endif
