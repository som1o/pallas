#ifndef SIMULATION_ENGINE_H
#define SIMULATION_ENGINE_H

#include <cstdint>
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
    Fixed technology_level = Fixed::from_int(60);
    Fixed resource_reserve = Fixed::from_int(70);
    Fixed supply_level = Fixed::from_int(70);
    Fixed supply_capacity = Fixed::from_int(72);
    Fixed trade_balance = Fixed::from_int(0);
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
    std::vector<uint16_t> adjacent_country_ids;
    std::vector<uint16_t> allied_country_ids;
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
    GridMap() = default;
    GridMap(uint32_t width, uint32_t height);

    uint32_t width() const;
    uint32_t height() const;

    uint16_t at(uint32_t x, uint32_t y) const;
    void set(uint32_t x, uint32_t y, uint16_t country_id);

    const std::vector<uint16_t>& flattened_country_ids() const;
    std::vector<uint16_t>& flattened_country_ids();

    bool save_binary(const std::string& path) const;
    bool load_binary(const std::string& path);

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::vector<uint16_t> country_ids_;
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
    NegotiationEvent(uint16_t country_a, uint16_t country_b);

    void execute(World& world) override;

private:
    uint16_t country_a_;
    uint16_t country_b_;
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
                                bool allow_nuclear = true);
    CombatResult resolve_offensive_tick(uint16_t attacker_id,
                                        uint16_t defender_id,
                                        uint32_t route_distance,
                                        uint32_t remaining_ticks);
    void resolve_negotiation(uint16_t country_a, uint16_t country_b);

private:
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
    bool has_active_trade_link(const Country& a, uint16_t other_id) const;
    bool is_embargoed_between(const Country& a, const Country& b) const;
    void update_trade_and_internal_politics();
    void update_country_dynamics_parallel();
    void update_opponent_models();
    void prune_recent_betrayal_memory(Country& country);

    std::vector<Country> countries_;
    std::unordered_map<uint16_t, size_t> country_index_by_id_;
    GridMap map_;

    uint64_t current_tick_ = 0;
    uint64_t tick_seconds_ = 3600;
    uint64_t base_seed_ = 1;
    uint64_t next_event_sequence_ = 0;

    std::priority_queue<ScheduledEvent, std::vector<ScheduledEvent>, ScheduledEventCompare> events_;
    std::vector<uint64_t> random_seed_log_;
};

}  // namespace sim

#endif