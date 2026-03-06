#include "strategy_utils.h"

#include "common_utils.h"

namespace pallas {
namespace strategy {

const std::unordered_map<std::string, Strategy>& strategy_lookup() {
    static const std::unordered_map<std::string, Strategy> kLookup = {
        {"attack", Strategy::Attack},
        {"defend", Strategy::Defend},
        {"negotiate", Strategy::Negotiate},
        {"surrender", Strategy::Surrender},
        {"transfer_weapons", Strategy::TransferWeapons},
        {"focus_economy", Strategy::FocusEconomy},
        {"develop_technology", Strategy::DevelopTechnology},
        {"form_alliance", Strategy::FormAlliance},
        {"betray", Strategy::Betray},
        {"cyber_operation", Strategy::CyberOperation},
        {"sign_trade_agreement", Strategy::SignTradeAgreement},
        {"cancel_trade_agreement", Strategy::CancelTradeAgreement},
        {"impose_embargo", Strategy::ImposeEmbargo},
        {"invest_in_resource_extraction", Strategy::InvestInResourceExtraction},
        {"reduce_military_upkeep", Strategy::ReduceMilitaryUpkeep},
        {"suppress_dissent", Strategy::SuppressDissent},
        {"hold_elections", Strategy::HoldElections},
        {"coup_attempt", Strategy::CoupAttempt},
        {"propose_defense_pact", Strategy::ProposeDefensePact},
        {"propose_non_aggression", Strategy::ProposeNonAggression},
        {"break_treaty", Strategy::BreakTreaty},
        {"request_intel", Strategy::RequestIntel},
        {"deploy_units", Strategy::DeployUnits},
        {"tactical_nuke", Strategy::TacticalNuke},
        {"strategic_nuke", Strategy::StrategicNuke},
        {"cyber_attack", Strategy::CyberAttack},
    };
    return kLookup;
}

std::optional<Strategy> strategy_from_string(const std::string& value) {
    const std::string lowered = util::to_lower_ascii(value);
    const auto& lookup = strategy_lookup();
    const auto it = lookup.find(lowered);
    if (it == lookup.end()) {
        return std::nullopt;
    }
    return it->second;
}

const std::unordered_map<std::string, uint32_t>& action_lookup() {
    static const std::unordered_map<std::string, uint32_t> kLookup = {
        {"attack", battle_common::kActionAttack},
        {"defend", battle_common::kActionDefend},
        {"negotiate", battle_common::kActionNegotiate},
        {"surrender", battle_common::kActionSurrender},
        {"transfer_weapons", battle_common::kActionTransferWeapons},
        {"focus_economy", battle_common::kActionFocusEconomy},
        {"develop_technology", battle_common::kActionDevelopTechnology},
        {"form_alliance", battle_common::kActionFormAlliance},
        {"betray", battle_common::kActionBetray},
        {"cyber_operation", battle_common::kActionCyberOperation},
        {"sign_trade_agreement", battle_common::kActionSignTradeAgreement},
        {"cancel_trade_agreement", battle_common::kActionCancelTradeAgreement},
        {"impose_embargo", battle_common::kActionImposeEmbargo},
        {"invest_in_resource_extraction", battle_common::kActionInvestInResourceExtraction},
        {"reduce_military_upkeep", battle_common::kActionReduceMilitaryUpkeep},
        {"suppress_dissent", battle_common::kActionSuppressDissent},
        {"hold_elections", battle_common::kActionHoldElections},
        {"coup_attempt", battle_common::kActionCoupAttempt},
        {"propose_defense_pact", battle_common::kActionProposeDefensePact},
        {"propose_non_aggression", battle_common::kActionProposeNonAggression},
        {"break_treaty", battle_common::kActionBreakTreaty},
        {"request_intel", battle_common::kActionRequestIntel},
        {"deploy_units", battle_common::kActionDeployUnits},
        {"tactical_nuke", battle_common::kActionTacticalNuke},
        {"strategic_nuke", battle_common::kActionStrategicNuke},
        {"cyber_attack", battle_common::kActionCyberAttack},
    };
    return kLookup;
}

uint32_t action_from_string(const std::string& value) {
    const std::string lowered = util::to_lower_ascii(value);
    const auto& lookup = action_lookup();
    const auto it = lookup.find(lowered);
    if (it == lookup.end()) {
        return battle_common::kActionDefend;
    }
    return it->second;
}

}  // namespace strategy
}  // namespace pallas
