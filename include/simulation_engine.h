#ifndef SIMULATION_ENGINE_H
#define SIMULATION_ENGINE_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sim {

class Fixed {
public:
    static constexpr int64_t kScale = 1000;

    Fixed();
    explicit Fixed(int64_t raw_value);

    static Fixed from_int(int64_t value);
    static Fixed from_milli(int64_t milli_value);
    static Fixed from_double(double value);

    int64_t raw() const;
    int64_t to_int() const;
    double to_double() const;

    Fixed operator+(const Fixed& other) const;
    Fixed operator-(const Fixed& other) const;
    Fixed operator*(const Fixed& other) const;
    Fixed operator/(const Fixed& other) const;
    Fixed& operator+=(const Fixed& other);
    Fixed& operator-=(const Fixed& other);

    bool operator<(const Fixed& other) const;
    bool operator>(const Fixed& other) const;
    bool operator<=(const Fixed& other) const;
    bool operator>=(const Fixed& other) const;
    bool operator==(const Fixed& other) const;

private:
    int64_t raw_;
};

struct Coordinate {
    int32_t x = 0;
    int32_t y = 0;
};

enum class DiplomaticStance : uint8_t {
    Aggressive = 0,
    Neutral = 1,
    Pacifist = 2,
};

enum class InternalUnrestStage : uint8_t {
    Calm = 0,
    Protests = 1,
    Strikes = 2,
    Riots = 3,
    CivilWar = 4,
};

enum class RegimeType : uint8_t {
    Democratic = 0,
    Hybrid = 1,
    Authoritarian = 2,
};

struct LeaderTraits {
    Fixed aggressive = Fixed::from_int(50);
    Fixed diplomatic = Fixed::from_int(50);
    Fixed corrupt = Fixed::from_int(40);
    Fixed competent = Fixed::from_int(50);
};

struct MilitaryPower {
    Fixed units_infantry = Fixed::from_int(220);
    Fixed units_armor = Fixed::from_int(40);
    Fixed units_artillery = Fixed::from_int(32);
    Fixed units_air_fighter = Fixed::from_int(18);
    Fixed units_air_bomber = Fixed::from_int(12);
    Fixed units_naval_surface = Fixed::from_int(16);
    Fixed units_naval_submarine = Fixed::from_int(8);
    Fixed ground_speed = Fixed::from_int(55);
    Fixed armor_speed = Fixed::from_int(50);
    Fixed air_sortie_rate = Fixed::from_int(60);

    Fixed ground_total() const;
    Fixed air_total() const;
    Fixed naval_total() const;
    Fixed weighted_total() const;
};

struct TerrainProfile {
    Fixed mountains = Fixed::from_double(0.25);
    Fixed forests = Fixed::from_double(0.30);
    Fixed urban = Fixed::from_double(0.20);
};

struct TechnologyTree {
    Fixed missile_defense = Fixed::from_int(50);
    Fixed cyber_warfare = Fixed::from_int(50);
    Fixed electronic_warfare = Fixed::from_int(50);
    Fixed drone_operations = Fixed::from_int(50);
};

struct ResourceDiversity {
    Fixed oil = Fixed::from_int(50);
    Fixed minerals = Fixed::from_int(50);
    Fixed food = Fixed::from_int(50);
    Fixed rare_earth = Fixed::from_int(40);
};

struct InternalPolitics {
    Fixed government_stability = Fixed::from_int(65);
    Fixed public_dissent = Fixed::from_int(20);
    Fixed corruption = Fixed::from_int(25);
};

struct Country {
    uint16_t id = 0;
    std::string name;
    std::string color;
    std::vector<Coordinate> shape;
    Coordinate capital;
    uint64_t population = 0;
    MilitaryPower military;
    Fixed economic_stability = Fixed::from_int(100);
    Fixed civilian_morale = Fixed::from_int(100);
    Fixed logistics_capacity = Fixed::from_int(70);
    Fixed intelligence_level = Fixed::from_int(70);
    Fixed industrial_output = Fixed::from_int(70);
    Fixed industrial_capital = Fixed::from_int(84);
    Fixed labor_participation = Fixed::from_double(0.58);
    Fixed technology_multiplier = Fixed::from_double(1.0);
    Fixed gdp_output = Fixed::from_int(70);
    Fixed technology_level = Fixed::from_int(60);
    Fixed resource_reserve = Fixed::from_int(70);
    Fixed supply_level = Fixed::from_int(70);
    Fixed supply_capacity = Fixed::from_int(72);
    Fixed trade_balance = Fixed::from_int(0);
    Fixed import_price_index = Fixed::from_double(1.0);
    std::vector<uint16_t> trade_partners;
    std::vector<uint16_t> embargoed_country_ids;
    std::vector<uint16_t> has_defense_pact_with;
    std::vector<uint16_t> has_non_aggression_with;
    std::vector<uint16_t> has_trade_treaty_with;
    Fixed resource_oil_reserves = Fixed::from_int(75);
    Fixed resource_minerals_reserves = Fixed::from_int(70);
    Fixed resource_food_reserves = Fixed::from_int(72);
    Fixed resource_rare_earth_reserves = Fixed::from_int(55);
    Fixed military_upkeep = Fixed::from_int(18);
    Fixed faction_military = Fixed::from_int(34);
    Fixed faction_industrial = Fixed::from_int(33);
    Fixed faction_civilian = Fixed::from_int(33);
    Fixed faction_hawks = Fixed::from_int(28);
    Fixed faction_economic_liberals = Fixed::from_int(26);
    Fixed faction_nationalists = Fixed::from_int(24);
    Fixed faction_populists = Fixed::from_int(22);
    Fixed war_economy_intensity = Fixed::from_int(18);
    Fixed industrial_decay = Fixed::from_int(0);
    Fixed debt_to_gdp = Fixed::from_int(22);
    Fixed war_bond_stock = Fixed::from_int(0);
    Fixed coup_risk = Fixed::from_int(8);
    int election_cycle = 12;
    Fixed draft_level = Fixed::from_int(20);
    Fixed war_weariness = Fixed::from_int(10);
    Fixed weather_severity = Fixed::from_int(30);
    Fixed seasonal_effect = Fixed::from_int(50);
    TerrainProfile terrain;
    TechnologyTree technology;
    ResourceDiversity resources;
    InternalPolitics politics;
    Fixed supply_stockpile = Fixed::from_int(70);
    Fixed nuclear_readiness = Fixed::from_int(40);
    Fixed deterrence_posture = Fixed::from_int(45);
    Fixed reputation = Fixed::from_int(55);
    Fixed escalation_level = Fixed::from_int(0);
    bool second_strike_capable = false;
    DiplomaticStance diplomatic_stance = DiplomaticStance::Neutral;
    InternalUnrestStage unrest_stage = InternalUnrestStage::Calm;
    RegimeType regime_type = RegimeType::Hybrid;
    LeaderTraits leader_traits;
    uint32_t leader_tenure_ticks = 0;
    std::vector<uint16_t> adjacent_country_ids;
    std::vector<uint16_t> allied_country_ids;
    uint16_t coalition_id = 0;
    std::unordered_map<uint16_t, Fixed> trust_scores;
    std::unordered_map<uint16_t, Fixed> intel_on_enemy;
    std::unordered_map<uint16_t, Fixed> believed_army_size;
    std::unordered_map<uint16_t, Fixed> opponent_model_confidence;
    std::unordered_map<uint16_t, uint64_t> defense_pact_expiry_ticks;
    std::unordered_map<uint16_t, uint64_t> non_aggression_expiry_ticks;
    std::unordered_map<uint16_t, uint64_t> trade_treaty_expiry_ticks;
    uint32_t territory_cells = 0;
    Fixed recent_combat_losses = Fixed::from_int(0);
    uint32_t war_duration_ticks = 0;
    Fixed strategic_depth = Fixed::from_int(2);
    std::vector<uint64_t> betrayal_tick_log;
};

class GridMap {
public:
    static constexpr uint8_t kTagSea = 1U << 0U;
    static constexpr uint8_t kTagStrategic = 1U << 1U;
    static constexpr uint8_t kTagChokepointStrait = 1U << 2U;
    static constexpr uint8_t kTagChokepointCanal = 1U << 3U;
    static constexpr uint8_t kTagMountainPass = 1U << 4U;
    static constexpr uint8_t kTagRiverCrossing = 1U << 5U;
    static constexpr uint8_t kTagPort = 1U << 6U;

    GridMap() = default;
    GridMap(uint32_t width, uint32_t height);

    uint32_t width() const;
    uint32_t height() const;

    uint16_t at(uint32_t x, uint32_t y) const;
    void set(uint32_t x, uint32_t y, uint16_t country_id);
    uint8_t cell_tags_at(uint32_t x, uint32_t y) const;
    void set_cell_tags(uint32_t x, uint32_t y, uint8_t tags);
    void add_cell_tag(uint32_t x, uint32_t y, uint8_t tag);
    uint16_t sea_zone_at(uint32_t x, uint32_t y) const;
    void set_sea_zone(uint32_t x, uint32_t y, uint16_t zone_id);
    bool is_sea_cell(uint32_t x, uint32_t y) const;
    bool has_tag(uint32_t x, uint32_t y, uint8_t tag) const;

    const std::vector<uint16_t>& flattened_country_ids() const;
    std::vector<uint16_t>& flattened_country_ids();
    const std::vector<uint8_t>& flattened_cell_tags() const;
    std::vector<uint8_t>& flattened_cell_tags();
    const std::vector<uint16_t>& flattened_sea_zone_ids() const;
    std::vector<uint16_t>& flattened_sea_zone_ids();

    bool save_binary(const std::string& path) const;
    bool load_binary(const std::string& path);

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::vector<uint16_t> country_ids_;
    std::vector<uint8_t> cell_tags_;
    std::vector<uint16_t> sea_zone_ids_;
};

struct CombatResult {
    Fixed attacker_army_losses;
    Fixed attacker_navy_losses;
    Fixed attacker_air_losses;
    Fixed attacker_missile_losses;
    Fixed defender_army_losses;
    Fixed defender_navy_losses;
    Fixed defender_air_losses;
    Fixed defender_missile_losses;
    int32_t territory_delta = 0;
    int32_t population_displaced = 0;
    Fixed infrastructure_damage;
    Fixed diplomatic_shock;
    bool nuclear_exchange = false;
    bool offensive_continues = false;
    uint64_t rng_seed_used = 0;
};

struct SupplyRouteAssessment {
    uint32_t route_distance = 1;
    double efficiency = 1.0;
    double interdiction = 0.0;
    bool encircled = false;
};

class World;

class Event {
public:
    virtual ~Event() = default;
    virtual void execute(World& world) = 0;
};

class AttackEvent final : public Event {
public:
    AttackEvent(uint16_t attacker_id,
                uint16_t defender_id,
                Fixed terrain_factor,
                Fixed surprise_factor,
                uint32_t route_distance = 1);

    void execute(World& world) override;

private:
    uint16_t attacker_id_;
    uint16_t defender_id_;
    Fixed terrain_factor_;
    Fixed surprise_factor_;
    uint32_t route_distance_;
};

class OffensiveEvent final : public Event {
public:
    OffensiveEvent(uint16_t attacker_id,
                   uint16_t defender_id,
                   uint32_t remaining_ticks,
                   uint32_t route_distance = 1);

    void execute(World& world) override;

private:
    uint16_t attacker_id_;
    uint16_t defender_id_;
    uint32_t remaining_ticks_;
    uint32_t route_distance_;
};

class NegotiationEvent final : public Event {
public:
    NegotiationEvent(uint16_t country_a,
                     uint16_t country_b,
                     uint16_t proposer_id = 0,
                     std::string terms_type = "ceasefire",
                     std::string terms_details = "");

    void execute(World& world) override;

private:
    uint16_t country_a_;
    uint16_t country_b_;
    uint16_t proposer_id_;
    std::string terms_type_;
    std::string terms_details_;
};

class World {
public:
    explicit World(uint64_t random_seed = 1, uint64_t tick_seconds = 3600);

    void set_map(const GridMap& map);
    GridMap& mutable_map();
    const GridMap& map() const;

    void add_country(const Country& country);
    std::vector<Country>& mutable_countries();
    const std::vector<Country>& countries() const;

    void schedule_event(std::unique_ptr<Event> event, uint64_t execute_at_tick);

    void run_tick();
    void run_ticks(uint64_t ticks);

    uint64_t current_tick() const;
    uint64_t tick_seconds() const;
    const std::vector<uint64_t>& random_seed_log() const;

    CombatResult resolve_attack(uint16_t attacker_id,
                                uint16_t defender_id,
                                Fixed terrain_factor,
                                Fixed surprise_factor,
                                uint32_t route_distance = 1,
                                bool allow_nuclear = true,
                                double frontline_momentum = 0.0,
                                double attacker_exhaustion = 0.0,
                                double defender_exhaustion = 0.0,
                                double reinforcement_factor = 1.0);
    CombatResult resolve_offensive_tick(uint16_t attacker_id,
                                        uint16_t defender_id,
                                        uint32_t route_distance,
                                        uint32_t remaining_ticks);
    void resolve_negotiation(uint16_t country_a, uint16_t country_b);
    void resolve_negotiation(uint16_t country_a,
                             uint16_t country_b,
                             uint16_t proposer_id,
                             const std::string& terms_type,
                             const std::string& terms_details);

private:
    struct ActiveEngagement {
        uint32_t remaining_ticks = 0;
        uint32_t route_distance = 1;
        double momentum = 0.0;
        double attacker_exhaustion = 0.0;
        double defender_exhaustion = 0.0;
    };

    struct NegotiationSession {
        uint16_t country_a = 0;
        uint16_t country_b = 0;
        uint16_t proposer_id = 0;
        uint8_t round = 0;
        uint8_t stage = 0;
        std::string terms_type;
        std::string terms_details;
        uint64_t last_updated_tick = 0;
    };

    struct ScheduledEvent {
        uint64_t tick = 0;
        uint64_t sequence = 0;
        std::shared_ptr<Event> event;
    };

    struct ScheduledEventCompare {
        bool operator()(const ScheduledEvent& lhs, const ScheduledEvent& rhs) const;
    };

    Country* find_country(uint16_t id);
    const Country* find_country(uint16_t id) const;
    void rebuild_country_index();

    void recompute_territory_cells();
    int32_t transfer_border_cells(uint16_t attacker_id, uint16_t defender_id, uint32_t max_cells);
    uint64_t derive_event_seed(uint16_t attacker_id, uint16_t defender_id) const;
    bool countries_share_land_frontier(uint16_t country_a, uint16_t country_b) const;
    std::vector<uint16_t> coastal_sea_zones(uint16_t country_id) const;
    void recompute_naval_control();
    uint16_t dominant_controller_for_zone(uint16_t zone_id, double* dominance_ratio = nullptr) const;
    bool can_launch_amphibious_assault(uint16_t attacker_id, uint16_t defender_id) const;
    double strategic_frontier_bonus(uint16_t owner_id, uint16_t against_id) const;
    double strategic_economic_bonus(uint16_t owner_id) const;
    double blockade_pressure_against(uint16_t country_id) const;
    bool has_hostile_relationship(uint16_t controller_id, uint16_t target_id) const;
    bool has_active_trade_link(const Country& a, uint16_t other_id) const;
    bool is_embargoed_between(const Country& a, const Country& b) const;
    void update_trade_and_internal_politics();
    void update_diplomatic_trust_and_pacts();
    void update_country_dynamics_parallel();
    void update_opponent_models();
    void prune_recent_betrayal_memory(Country& country);
    Fixed bilateral_trust(const Country& from, uint16_t other_id) const;
    Fixed bilateral_trust_floor(const Country& a, const Country& b) const;
    void enforce_pact_gates(Country* a, Country* b);
    void register_betrayal(uint16_t betrayer_id, uint16_t victim_id, Fixed severity);
    double betrayal_decay_score(uint16_t betrayer_id) const;
    void align_coalition_allies();
    void merge_or_create_coalition(uint16_t country_a, uint16_t country_b);
    uint32_t negotiation_key(uint16_t country_a, uint16_t country_b) const;
    void ensure_map_state_buffers();
    void update_fortification_levels();
    double frontier_fortification_bonus(uint16_t owner_id, uint16_t against_id) const;
    SupplyRouteAssessment assess_supply_route(uint16_t country_id, uint16_t enemy_id) const;
    uint64_t engagement_key(uint16_t attacker_id, uint16_t defender_id) const;

    std::vector<Country> countries_;
    std::unordered_map<uint16_t, size_t> country_index_by_id_;
    GridMap map_;

    uint64_t current_tick_ = 0;
    uint64_t tick_seconds_ = 3600;
    uint64_t base_seed_ = 1;
    uint64_t next_event_sequence_ = 0;

    std::priority_queue<ScheduledEvent, std::vector<ScheduledEvent>, ScheduledEventCompare> events_;
    std::vector<uint64_t> random_seed_log_;
    std::vector<uint16_t> fortification_level_per_cell_;
    std::vector<uint16_t> previous_owner_per_cell_;
    std::unordered_map<uint16_t, uint16_t> dominant_controller_by_sea_zone_;
    std::unordered_map<uint16_t, double> naval_dominance_ratio_by_sea_zone_;
    std::unordered_map<uint16_t, double> blockade_pressure_by_country_;
    std::unordered_map<uint16_t, double> strategic_economic_bonus_by_country_;
    std::unordered_map<uint64_t, ActiveEngagement> active_engagements_;
    std::unordered_map<uint32_t, NegotiationSession> negotiation_sessions_;
    std::unordered_map<uint16_t, std::vector<uint64_t>> betrayal_history_by_betrayer_;
    uint16_t next_coalition_id_ = 1;
};

}  // namespace sim

#endif