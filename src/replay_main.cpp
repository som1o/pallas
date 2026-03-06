#include "battle_runtime.h"

#include <iostream>
#include <string>

namespace {

const char* strategy_to_string(Strategy strategy) {
    switch (strategy) {
        case Strategy::Attack: return "attack";
        case Strategy::Defend: return "defend";
        case Strategy::Negotiate: return "negotiate";
        case Strategy::Surrender: return "surrender";
        case Strategy::TransferWeapons: return "transfer_weapons";
        case Strategy::FocusEconomy: return "focus_economy";
        case Strategy::DevelopTechnology: return "develop_technology";
        case Strategy::FormAlliance: return "form_alliance";
        case Strategy::Betray: return "betray";
        case Strategy::CyberOperation: return "cyber_operation";
        case Strategy::SignTradeAgreement: return "sign_trade_agreement";
        case Strategy::CancelTradeAgreement: return "cancel_trade_agreement";
        case Strategy::ImposeEmbargo: return "impose_embargo";
        case Strategy::InvestInResourceExtraction: return "invest_in_resource_extraction";
        case Strategy::ReduceMilitaryUpkeep: return "reduce_military_upkeep";
        case Strategy::SuppressDissent: return "suppress_dissent";
        case Strategy::HoldElections: return "hold_elections";
        case Strategy::CoupAttempt: return "coup_attempt";
        case Strategy::ProposeDefensePact: return "propose_defense_pact";
        case Strategy::ProposeNonAggression: return "propose_non_aggression";
        case Strategy::BreakTreaty: return "break_treaty";
        case Strategy::RequestIntel: return "request_intel";
        case Strategy::DeployUnits: return "deploy_units";
        case Strategy::TacticalNuke: return "tactical_nuke";
        case Strategy::StrategicNuke: return "strategic_nuke";
        case Strategy::CyberAttack: return "cyber_attack";
    }
    return "defend";
}

}  // namespace

int main(int argc, char** argv) {
    std::string path = "../logs/battle_replay.bin";
    int limit = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--log" && i + 1 < argc) {
            path = argv[++i];
        } else if (arg == "--ticks" && i + 1 < argc) {
            limit = std::stoi(argv[++i]);
        }
    }

    battle::ReplayReader reader;
    if (!reader.open(path)) {
        std::cerr << "failed to open replay log: " << path << std::endl;
        return 1;
    }

    battle::ReplayFrame frame;
    int count = 0;
    while (reader.read_next(&frame)) {
        std::cout << "tick " << frame.tick << "\n";
        std::cout << "  countries:\n";
        for (const auto& country : frame.countries) {
            std::cout << "    [" << country.id << "] " << country.name
                      << " alignment=" << country.team
                      << " infantry=" << (country.units_infantry_milli / 1000)
                      << " armor=" << (country.units_armor_milli / 1000)
                      << " artillery=" << (country.units_artillery_milli / 1000)
                      << " morale=" << (country.civilian_morale_milli / 1000)
                      << " econ=" << (country.economic_stability_milli / 1000)
                      << " trade=" << (country.trade_balance_milli / 1000)
                      << " escalation=" << (country.escalation_level_milli / 1000)
                      << " coup_risk=" << (country.coup_risk_milli / 1000)
                      << " territory=" << country.territory_cells << "\n";
        }
        std::cout << "  decisions:\n";
        for (const auto& decision : frame.decisions) {
            std::cout << "    " << decision.model_name
                      << " actor=" << decision.decision.actor_country_id
                      << " strategy=" << strategy_to_string(decision.decision.strategy)
                      << " target=" << decision.decision.target_country_id;
            if (!decision.decision.terms.type.empty()) {
                std::cout << " terms=" << decision.decision.terms.type;
            }
            std::cout << "\n";
        }

        ++count;
        if (limit >= 0 && count >= limit) {
            break;
        }
    }

    return 0;
}
