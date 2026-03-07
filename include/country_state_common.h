#ifndef COUNTRY_STATE_COMMON_H
#define COUNTRY_STATE_COMMON_H

#include <cstdint>
#include <string>
#include <vector>

struct CountryStateCommon {
    uint16_t id = 0;
    std::string color;
    uint64_t population = 0;
    int64_t units_infantry_milli = 0;
    int64_t units_armor_milli = 0;
    int64_t units_artillery_milli = 0;
    int64_t units_air_fighter_milli = 0;
    int64_t units_air_bomber_milli = 0;
    int64_t units_naval_surface_milli = 0;
    int64_t units_naval_submarine_milli = 0;
    int64_t economic_stability_milli = 0;
    int64_t civilian_morale_milli = 0;
    int64_t logistics_milli = 0;
    int64_t intelligence_milli = 0;
    int64_t industry_milli = 0;
    int64_t industrial_capital_milli = 0;
    int64_t labor_participation_milli = 0;
    int64_t technology_multiplier_milli = 0;
    int64_t gdp_output_milli = 0;
    int64_t technology_milli = 0;
    int64_t resource_reserve_milli = 0;
    int64_t supply_level_milli = 0;
    int64_t supply_capacity_milli = 0;
    int64_t trade_balance_milli = 0;
    int64_t import_price_index_milli = 0;
    std::vector<uint16_t> trade_partner_ids;
    std::vector<uint16_t> defense_pact_ids;
    std::vector<uint16_t> non_aggression_pact_ids;
    std::vector<uint16_t> trade_treaty_ids;
    int64_t resource_oil_reserves_milli = 0;
    int64_t resource_minerals_reserves_milli = 0;
    int64_t resource_food_reserves_milli = 0;
    int64_t resource_rare_earth_reserves_milli = 0;
    int64_t military_upkeep_milli = 0;
    int64_t faction_military_milli = 0;
    int64_t faction_industrial_milli = 0;
    int64_t faction_civilian_milli = 0;
    int64_t war_economy_intensity_milli = 0;
    int64_t industrial_decay_milli = 0;
    int64_t debt_to_gdp_milli = 0;
    int64_t war_bond_stock_milli = 0;
    int64_t coup_risk_milli = 0;
    int32_t election_cycle = 0;
    int64_t draft_level_milli = 0;
    int64_t war_weariness_milli = 0;
    int64_t reputation_milli = 0;
    int64_t escalation_level_milli = 0;
    int32_t recent_betrayals = 0;
    uint8_t diplomatic_stance = 1;
    bool second_strike_capable = false;
    std::vector<uint16_t> adjacent_country_ids;
    std::vector<uint16_t> allied_country_ids;
};

#endif
