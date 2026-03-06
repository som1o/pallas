#include "simulation_engine.h"
#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

#include <omp.h>

namespace sim {

namespace {

constexpr uint32_t kMapMagic = 0x50414C4D;
constexpr uint32_t kMapVersion = 1;

uint64_t mix_u64(uint64_t value) {
    value ^= value >> 33U;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33U;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33U;
    return value;
}

Fixed clamp_fixed(const Fixed& value, const Fixed& lo, const Fixed& hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

Fixed country_strength_estimate(const Country& country) {
    return country.military.weighted_total();
}

double normalized_event_roll(uint64_t seed, uint64_t salt) {
    const uint64_t mixed = mix_u64(seed ^ mix_u64(salt));
    return static_cast<double>(mixed % 10000ULL) / 10000.0;
}

}  // namespace

Fixed::Fixed() : raw_(0) {}

Fixed::Fixed(int64_t raw_value) : raw_(raw_value) {}

Fixed Fixed::from_int(int64_t value) {
    return Fixed(value * kScale);
}

Fixed Fixed::from_milli(int64_t milli_value) {
    return Fixed(milli_value);
}

Fixed Fixed::from_double(double value) {
    return Fixed(static_cast<int64_t>(std::llround(value * static_cast<double>(kScale))));
}

int64_t Fixed::raw() const {
    return raw_;
}

int64_t Fixed::to_int() const {
    return raw_ / kScale;
}

double Fixed::to_double() const {
    return static_cast<double>(raw_) / static_cast<double>(kScale);
}

Fixed Fixed::operator+(const Fixed& other) const {
    return Fixed(raw_ + other.raw_);
}

Fixed Fixed::operator-(const Fixed& other) const {
    return Fixed(raw_ - other.raw_);
}

Fixed Fixed::operator*(const Fixed& other) const {
#if defined(__SIZEOF_INT128__)
    const __int128 product = static_cast<__int128>(raw_) * static_cast<__int128>(other.raw_);
    const __int128 scaled = product / static_cast<__int128>(kScale);
    if (scaled > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        return Fixed(std::numeric_limits<int64_t>::max());
    }
    if (scaled < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
        return Fixed(std::numeric_limits<int64_t>::min());
    }
    return Fixed(static_cast<int64_t>(scaled));
#else
    const long double scaled =
        (static_cast<long double>(raw_) * static_cast<long double>(other.raw_)) / static_cast<long double>(kScale);
    if (scaled > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return Fixed(std::numeric_limits<int64_t>::max());
    }
    if (scaled < static_cast<long double>(std::numeric_limits<int64_t>::min())) {
        return Fixed(std::numeric_limits<int64_t>::min());
    }
    return Fixed(static_cast<int64_t>(scaled));
#endif
}

Fixed Fixed::operator/(const Fixed& other) const {
    if (other.raw_ == 0) {
        throw std::runtime_error("Fixed divide by zero");
    }
    return Fixed((raw_ * kScale) / other.raw_);
}

Fixed& Fixed::operator+=(const Fixed& other) {
    raw_ += other.raw_;
    return *this;
}

Fixed& Fixed::operator-=(const Fixed& other) {
    raw_ -= other.raw_;
    return *this;
}

bool Fixed::operator<(const Fixed& other) const {
    return raw_ < other.raw_;
}

bool Fixed::operator>(const Fixed& other) const {
    return raw_ > other.raw_;
}

bool Fixed::operator<=(const Fixed& other) const {
    return raw_ <= other.raw_;
}

bool Fixed::operator>=(const Fixed& other) const {
    return raw_ >= other.raw_;
}

bool Fixed::operator==(const Fixed& other) const {
    return raw_ == other.raw_;
}

Fixed MilitaryPower::ground_total() const {
    return units_infantry + units_armor + units_artillery;
}

Fixed MilitaryPower::air_total() const {
    return units_air_fighter + units_air_bomber;
}

Fixed MilitaryPower::naval_total() const {
    return units_naval_surface + units_naval_submarine;
}

Fixed MilitaryPower::weighted_total() const {
    const Fixed armor_weight = Fixed::from_double(1.55);
    const Fixed artillery_weight = Fixed::from_double(1.28);
    const Fixed fighter_weight = Fixed::from_double(1.38);
    const Fixed bomber_weight = Fixed::from_double(1.62);
    const Fixed surface_weight = Fixed::from_double(1.42);
    const Fixed submarine_weight = Fixed::from_double(1.56);
    const Fixed maneuver = (ground_speed + armor_speed + air_sortie_rate) / Fixed::from_int(300);
    return (units_infantry + (units_armor * armor_weight) + (units_artillery * artillery_weight) +
            (units_air_fighter * fighter_weight) + (units_air_bomber * bomber_weight) +
            (units_naval_surface * surface_weight) + (units_naval_submarine * submarine_weight)) *
           (Fixed::from_double(0.85) + maneuver);
}

GridMap::GridMap(uint32_t width, uint32_t height)
    : width_(width), height_(height), country_ids_(static_cast<size_t>(width) * static_cast<size_t>(height), 0) {}

uint32_t GridMap::width() const {
    return width_;
}

uint32_t GridMap::height() const {
    return height_;
}

uint16_t GridMap::at(uint32_t x, uint32_t y) const {
    return country_ids_.at(static_cast<size_t>(y) * width_ + x);
}

void GridMap::set(uint32_t x, uint32_t y, uint16_t country_id) {
    country_ids_.at(static_cast<size_t>(y) * width_ + x) = country_id;
}

const std::vector<uint16_t>& GridMap::flattened_country_ids() const {
    return country_ids_;
}

std::vector<uint16_t>& GridMap::flattened_country_ids() {
    return country_ids_;
}

bool GridMap::save_binary(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    pallas::util::write_binary(out, kMapMagic);
    pallas::util::write_binary(out, kMapVersion);
    pallas::util::write_binary(out, width_);
    pallas::util::write_binary(out, height_);
    const uint64_t cells = static_cast<uint64_t>(country_ids_.size());
    pallas::util::write_binary(out, cells);
    if (!country_ids_.empty()) {
        out.write(reinterpret_cast<const char*>(country_ids_.data()), static_cast<std::streamsize>(country_ids_.size() * sizeof(uint16_t)));
    }
    return static_cast<bool>(out);
}

bool GridMap::load_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t cells = 0;
    if (!pallas::util::read_binary(in, &magic) || !pallas::util::read_binary(in, &version) ||
        !pallas::util::read_binary(in, &width_) || !pallas::util::read_binary(in, &height_) ||
        !pallas::util::read_binary(in, &cells)) {
        return false;
    }
    if (magic != kMapMagic || version != kMapVersion) {
        return false;
    }
    const uint64_t expected_cells = static_cast<uint64_t>(width_) * static_cast<uint64_t>(height_);
    if (cells != expected_cells || cells > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    country_ids_.resize(static_cast<size_t>(cells));
    if (!country_ids_.empty()) {
        in.read(reinterpret_cast<char*>(country_ids_.data()), static_cast<std::streamsize>(country_ids_.size() * sizeof(uint16_t)));
    }
    return static_cast<bool>(in);
}

AttackEvent::AttackEvent(uint16_t attacker_id,
                         uint16_t defender_id,
                         Fixed terrain_factor,
                                                 Fixed surprise_factor,
                                                 uint32_t route_distance)
    : attacker_id_(attacker_id),
      defender_id_(defender_id),
      terrain_factor_(terrain_factor),
            surprise_factor_(surprise_factor),
            route_distance_(route_distance) {}

void AttackEvent::execute(World& world) {
        world.resolve_attack(attacker_id_, defender_id_, terrain_factor_, surprise_factor_, route_distance_);
}

OffensiveEvent::OffensiveEvent(uint16_t attacker_id,
                                                             uint16_t defender_id,
                                                             uint32_t remaining_ticks,
                                                             uint32_t route_distance)
        : attacker_id_(attacker_id),
            defender_id_(defender_id),
            remaining_ticks_(remaining_ticks),
            route_distance_(route_distance) {}

void OffensiveEvent::execute(World& world) {
        world.resolve_offensive_tick(attacker_id_, defender_id_, route_distance_, remaining_ticks_);
}

NegotiationEvent::NegotiationEvent(uint16_t country_a, uint16_t country_b)
    : country_a_(country_a), country_b_(country_b) {}

void NegotiationEvent::execute(World& world) {
    world.resolve_negotiation(country_a_, country_b_);
}

World::World(uint64_t random_seed, uint64_t tick_seconds)
    : tick_seconds_(tick_seconds == 0 ? 3600 : tick_seconds),
      base_seed_(random_seed == 0 ? 1 : random_seed) {}

void World::set_map(const GridMap& map) {
    map_ = map;
    recompute_territory_cells();
}

GridMap& World::mutable_map() {
    return map_;
}

const GridMap& World::map() const {
    return map_;
}

void World::add_country(const Country& country) {
    countries_.push_back(country);
    country_index_by_id_[country.id] = countries_.size() - 1;
}

std::vector<Country>& World::mutable_countries() {
    return countries_;
}

const std::vector<Country>& World::countries() const {
    return countries_;
}

void World::schedule_event(std::unique_ptr<Event> event, uint64_t execute_at_tick) {
    ScheduledEvent item;
    item.tick = execute_at_tick;
    item.sequence = next_event_sequence_++;
    item.event = std::shared_ptr<Event>(std::move(event));
    events_.push(std::move(item));
}

void World::run_tick() {
    while (!events_.empty() && events_.top().tick <= current_tick_) {
        ScheduledEvent item = events_.top();
        events_.pop();
        if (item.event) {
            item.event->execute(*this);
        }
    }

    update_country_dynamics_parallel();
    update_trade_and_internal_politics();
    update_opponent_models();
    ++current_tick_;
}

void World::run_ticks(uint64_t ticks) {
    for (uint64_t i = 0; i < ticks; ++i) {
        run_tick();
    }
}

uint64_t World::current_tick() const {
    return current_tick_;
}

uint64_t World::tick_seconds() const {
    return tick_seconds_;
}

const std::vector<uint64_t>& World::random_seed_log() const {
    return random_seed_log_;
}

CombatResult World::resolve_attack(uint16_t attacker_id,
                                   uint16_t defender_id,
                                   Fixed terrain_factor,
                                   Fixed surprise_factor,
                                   uint32_t route_distance,
                                   bool allow_nuclear) {
    CombatResult result;

    Country* attacker = find_country(attacker_id);
    Country* defender = find_country(defender_id);
    if (attacker == nullptr || defender == nullptr) {
        return result;
    }

    if (pallas::util::contains_id(attacker->allied_country_ids, defender_id) ||
        pallas::util::contains_id(attacker->has_non_aggression_with, defender_id)) {
        return result;
    }

    const uint64_t event_seed = derive_event_seed(attacker_id, defender_id);
    result.rng_seed_used = event_seed;
    random_seed_log_.push_back(event_seed);
    std::mt19937_64 rng(event_seed);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

    const auto intel_value = [](const Country& country, uint16_t other_id) {
        const auto it = country.intel_on_enemy.find(other_id);
        if (it != country.intel_on_enemy.end()) {
            return std::clamp(it->second.to_double(), 0.0, 100.0);
        }
        return std::clamp(country.intelligence_level.to_double() * 0.75, 0.0, 100.0);
    };
    const auto combined_arms_bonus = [](const MilitaryPower& military) {
        int active_domains = 0;
        if (military.ground_total() > Fixed::from_int(60)) {
            ++active_domains;
        }
        if (military.air_total() > Fixed::from_int(10)) {
            ++active_domains;
        }
        if (military.naval_total() > Fixed::from_int(8)) {
            ++active_domains;
        }
        return 1.0 + static_cast<double>(std::max(0, active_domains - 1)) * 0.10;
    };
    const auto pact_support = [&](const Country& country, uint16_t against_id) {
        double support = 0.0;
        for (uint16_t supporter_id : country.has_defense_pact_with) {
            if (supporter_id == against_id) {
                continue;
            }
            const Country* supporter = find_country(supporter_id);
            if (supporter == nullptr) {
                continue;
            }
            support += supporter->military.weighted_total().to_double() * 0.14;
        }
        return support;
    };
    const auto ground_strength = [&](const Country& country, const Country& enemy, double rough_terrain, double open_terrain, double urban_terrain) {
        const double infantry = country.military.units_infantry.to_double() * (1.05 + rough_terrain * 0.16 + urban_terrain * 0.10);
        const double armor = country.military.units_armor.to_double() * (1.10 + open_terrain * 0.18 - rough_terrain * 0.28);
        const double artillery = country.military.units_artillery.to_double() * (1.08 + open_terrain * 0.20 - rough_terrain * 0.22);
        const double armor_vs_infantry = 1.0 + std::min(0.26, enemy.military.units_infantry.to_double() / std::max(350.0, country.military.units_armor.to_double() * 8.0));
        const double infantry_vs_artillery = 1.0 + rough_terrain * std::min(0.22, enemy.military.units_artillery.to_double() / std::max(280.0, country.military.units_infantry.to_double() * 3.5));
        const double artillery_vs_mass = 1.0 + open_terrain * std::min(0.18, enemy.military.units_infantry.to_double() / 420.0);
        return infantry * infantry_vs_artillery + armor * armor_vs_infantry + artillery * artillery_vs_mass;
    };
    const auto air_strength = [&](const Country& country, const Country& enemy) {
        const double fighters = country.military.units_air_fighter.to_double() * (1.08 + country.technology.electronic_warfare.to_double() / 240.0);
        const double bombers = country.military.units_air_bomber.to_double() * (1.12 + country.technology.drone_operations.to_double() / 240.0);
        const double fighter_screen = 1.0 + std::min(0.24, enemy.military.units_air_bomber.to_double() / std::max(60.0, country.military.units_air_fighter.to_double() * 2.0));
        const double bomber_pressure = 1.0 + std::min(0.18, enemy.military.units_armor.to_double() / 180.0);
        return fighters * fighter_screen + bombers * bomber_pressure;
    };
    const auto naval_strength = [&](const Country& country, const Country& enemy) {
        const double surface = country.military.units_naval_surface.to_double() * (1.08 + country.resources.oil.to_double() / 280.0);
        const double submarine = country.military.units_naval_submarine.to_double() * (1.16 + country.intelligence_level.to_double() / 260.0);
        const double sub_vs_surface = 1.0 + std::min(0.24, enemy.military.units_naval_surface.to_double() / std::max(45.0, country.military.units_naval_submarine.to_double() * 2.0));
        return surface + submarine * sub_vs_surface;
    };

    const double weather = 0.75 + 0.40 * unit_dist(rng);
    const double crit = unit_dist(rng) < 0.05 ? 1.15 : 1.0;
    const double attacker_observation = intel_value(*attacker, defender_id);
    const double defender_observation = intel_value(*defender, attacker_id);
    const double attacker_intel = 0.80 + std::min(0.55, attacker_observation / 180.0);
    const double defender_intel = 0.80 + std::min(0.55, defender_observation / 180.0);
    const double fog_of_war = std::clamp(0.82 + attacker_observation / 240.0 - defender_observation / 420.0, 0.72, 1.28);
    const double attacker_tech = 0.85 + std::min(0.50, attacker->technology_level.to_double() / 200.0);
    const double defender_tech = 0.85 + std::min(0.50, defender->technology_level.to_double() / 200.0);
    const double route_drag = std::max(0.58, 1.0 - static_cast<double>(route_distance) * 0.035);
    const double attacker_logistics = (0.80 + std::min(0.50, attacker->logistics_capacity.to_double() / 180.0)) * route_drag;
    const double defender_logistics = 0.80 + std::min(0.50, defender->logistics_capacity.to_double() / 180.0);
    const double attacker_supply = std::clamp(0.68 + attacker->supply_level.to_double() / 140.0 + attacker->supply_capacity.to_double() / 260.0, 0.45, 1.45);
    const double defender_supply = std::clamp(0.68 + defender->supply_level.to_double() / 140.0 + defender->supply_capacity.to_double() / 260.0, 0.45, 1.45);
    const double rough_terrain = std::clamp((defender->terrain.mountains.to_double() + defender->terrain.forests.to_double()) * 0.5, 0.0, 1.0);
    const double urban_terrain = std::clamp(defender->terrain.urban.to_double(), 0.0, 1.0);
    const double open_terrain = std::clamp(1.0 - rough_terrain - urban_terrain * 0.35, 0.2, 1.0);
    const double weather_penalty = 1.0 - std::min(0.35, (attacker->weather_severity.to_double() + defender->weather_severity.to_double()) / 600.0);
    const double season_effect = 0.90 + std::min(0.20, std::abs(attacker->seasonal_effect.to_double() - defender->seasonal_effect.to_double()) / 250.0);
    const double attack_strength =
        (ground_strength(*attacker, *defender, rough_terrain, open_terrain, urban_terrain) +
         air_strength(*attacker, *defender) + naval_strength(*attacker, *defender)) *
        terrain_factor.to_double() * surprise_factor.to_double() * fog_of_war * weather * crit * attacker_intel * attacker_tech *
        attacker_logistics * attacker_supply * combined_arms_bonus(attacker->military) * weather_penalty;
    const double defense_strength =
        ((ground_strength(*defender, *attacker, rough_terrain, open_terrain, urban_terrain) +
          air_strength(*defender, *attacker) + naval_strength(*defender, *attacker)) *
         (1.16 + defender->terrain.mountains.to_double() * 0.12 + urban_terrain * 0.10) + pact_support(*defender, attacker_id)) *
        (1.20 - std::min(0.80, terrain_factor.to_double() * 0.20)) * (1.0 + 0.10 * unit_dist(rng)) *
        defender_intel * defender_tech * defender_logistics * defender_supply * combined_arms_bonus(defender->military) * season_effect;

    const double total_strength = std::max(1e-6, attack_strength + defense_strength);
    const double intensity = 0.08 + 0.04 * unit_dist(rng);

    const double attacker_losses_ratio = (defense_strength / total_strength) * intensity;
    const double defender_losses_ratio = (attack_strength / total_strength) * intensity;

    const double attacker_attrition = std::min(0.24, static_cast<double>(route_distance) * 0.012 + (100.0 - attacker->supply_level.to_double()) / 420.0);
    const double defender_attrition = std::min(0.18, (100.0 - defender->supply_level.to_double()) / 580.0);

    const auto apply_unit_losses = [&](Country* country,
                                       double ratio,
                                       double attrition,
                                       Fixed* ground_losses,
                                       Fixed* naval_losses,
                                       Fixed* air_losses) {
        const double effective_ratio = std::max(0.0, ratio + attrition);
        const Fixed infantry_loss = Fixed::from_double(country->military.units_infantry.to_double() * effective_ratio * 0.42);
        const Fixed armor_loss = Fixed::from_double(country->military.units_armor.to_double() * effective_ratio * 0.22);
        const Fixed artillery_loss = Fixed::from_double(country->military.units_artillery.to_double() * effective_ratio * 0.18);
        const Fixed fighter_loss = Fixed::from_double(country->military.units_air_fighter.to_double() * effective_ratio * 0.21);
        const Fixed bomber_loss = Fixed::from_double(country->military.units_air_bomber.to_double() * effective_ratio * 0.17);
        const Fixed surface_loss = Fixed::from_double(country->military.units_naval_surface.to_double() * effective_ratio * 0.18);
        const Fixed submarine_loss = Fixed::from_double(country->military.units_naval_submarine.to_double() * effective_ratio * 0.14);

        country->military.units_infantry = clamp_fixed(country->military.units_infantry - infantry_loss, Fixed::from_int(0), Fixed::from_int(1000000));
        country->military.units_armor = clamp_fixed(country->military.units_armor - armor_loss, Fixed::from_int(0), Fixed::from_int(1000000));
        country->military.units_artillery = clamp_fixed(country->military.units_artillery - artillery_loss, Fixed::from_int(0), Fixed::from_int(1000000));
        country->military.units_air_fighter = clamp_fixed(country->military.units_air_fighter - fighter_loss, Fixed::from_int(0), Fixed::from_int(1000000));
        country->military.units_air_bomber = clamp_fixed(country->military.units_air_bomber - bomber_loss, Fixed::from_int(0), Fixed::from_int(1000000));
        country->military.units_naval_surface = clamp_fixed(country->military.units_naval_surface - surface_loss, Fixed::from_int(0), Fixed::from_int(1000000));
        country->military.units_naval_submarine = clamp_fixed(country->military.units_naval_submarine - submarine_loss, Fixed::from_int(0), Fixed::from_int(1000000));

        *ground_losses = infantry_loss + armor_loss + artillery_loss;
        *air_losses = fighter_loss + bomber_loss;
        *naval_losses = surface_loss + submarine_loss;
    };

    apply_unit_losses(attacker, attacker_losses_ratio, attacker_attrition,
                      &result.attacker_army_losses,
                      &result.attacker_navy_losses,
                      &result.attacker_air_losses);
    apply_unit_losses(defender, defender_losses_ratio, defender_attrition,
                      &result.defender_army_losses,
                      &result.defender_navy_losses,
                      &result.defender_air_losses);
    result.attacker_missile_losses = Fixed::from_int(0);
    result.defender_missile_losses = Fixed::from_int(0);

    attacker->economic_stability = clamp_fixed(attacker->economic_stability - Fixed::from_double(attacker_losses_ratio * 2.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->economic_stability = clamp_fixed(defender->economic_stability - Fixed::from_double(defender_losses_ratio * 3.0), Fixed::from_int(0), Fixed::from_int(100));

    attacker->civilian_morale = clamp_fixed(attacker->civilian_morale - Fixed::from_double(attacker_losses_ratio * 4.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->civilian_morale = clamp_fixed(defender->civilian_morale - Fixed::from_double(defender_losses_ratio * 6.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->resource_reserve = clamp_fixed(attacker->resource_reserve - Fixed::from_double(1.0 + attacker_losses_ratio * 12.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->resource_reserve = clamp_fixed(defender->resource_reserve - Fixed::from_double(1.0 + defender_losses_ratio * 12.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->logistics_capacity = clamp_fixed(attacker->logistics_capacity - Fixed::from_double(attacker_losses_ratio * 4.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->logistics_capacity = clamp_fixed(defender->logistics_capacity - Fixed::from_double(defender_losses_ratio * 5.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->supply_stockpile = clamp_fixed(attacker->supply_stockpile - Fixed::from_double(2.0 + attacker_attrition * 25.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->supply_stockpile = clamp_fixed(defender->supply_stockpile - Fixed::from_double(1.2 + defender_attrition * 20.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->supply_level = clamp_fixed(attacker->supply_level - Fixed::from_double(2.5 + attacker_attrition * 22.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->supply_level = clamp_fixed(defender->supply_level - Fixed::from_double(1.8 + defender_attrition * 18.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->recent_combat_losses += Fixed::from_double(attacker_losses_ratio * 100.0 + attacker_attrition * 20.0);
    defender->recent_combat_losses += Fixed::from_double(defender_losses_ratio * 100.0 + defender_attrition * 20.0);
    attacker->war_duration_ticks += 1;
    defender->war_duration_ticks += 1;
    attacker->war_weariness = clamp_fixed(attacker->war_weariness + Fixed::from_double(attacker_losses_ratio * 8.0 + 0.7), Fixed::from_int(0), Fixed::from_int(100));
    defender->war_weariness = clamp_fixed(defender->war_weariness + Fixed::from_double(defender_losses_ratio * 10.0 + 0.9), Fixed::from_int(0), Fixed::from_int(100));
    attacker->politics.public_dissent = clamp_fixed(attacker->politics.public_dissent + Fixed::from_double(attacker_losses_ratio * 4.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->politics.public_dissent = clamp_fixed(defender->politics.public_dissent + Fixed::from_double(defender_losses_ratio * 5.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->intel_on_enemy[defender_id] = clamp_fixed(attacker->intel_on_enemy[defender_id] + Fixed::from_double(3.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->intel_on_enemy[attacker_id] = clamp_fixed(defender->intel_on_enemy[attacker_id] + Fixed::from_double(4.0), Fixed::from_int(0), Fixed::from_int(100));
    pallas::util::erase_id(&attacker->trade_partners, defender_id);
    pallas::util::erase_id(&defender->trade_partners, attacker_id);

    result.infrastructure_damage = Fixed::from_double(std::min(45.0, defender_losses_ratio * 180.0 + attacker_losses_ratio * 60.0));
    result.population_displaced = static_cast<int32_t>(std::llround(result.infrastructure_damage.to_double() * 150.0));
    result.diplomatic_shock = Fixed::from_double(std::min(25.0, (attacker_losses_ratio + defender_losses_ratio) * 80.0));

    auto trust_it = attacker->trust_scores.find(defender_id);
    if (trust_it != attacker->trust_scores.end()) {
        trust_it->second = clamp_fixed(trust_it->second - result.diplomatic_shock / Fixed::from_int(8), Fixed::from_int(0), Fixed::from_int(100));
    }
    trust_it = defender->trust_scores.find(attacker_id);
    if (trust_it != defender->trust_scores.end()) {
        trust_it->second = clamp_fixed(trust_it->second - result.diplomatic_shock / Fixed::from_int(6), Fixed::from_int(0), Fixed::from_int(100));
    }

    if (allow_nuclear) {
        const double attacker_pressure = (100.0 - attacker->civilian_morale.to_double()) + (100.0 - attacker->politics.government_stability.to_double()) + attacker->escalation_level.to_double() * 10.0;
        const double defender_pressure = (100.0 - defender->civilian_morale.to_double()) + (100.0 - defender->politics.government_stability.to_double()) + defender->escalation_level.to_double() * 10.0;
        const double nuclear_threshold = 145.0;
        const bool attacker_can_nuclear = attacker->nuclear_readiness > Fixed::from_int(65) && attacker->deterrence_posture > Fixed::from_int(55);
        const bool defender_can_nuclear = defender->nuclear_readiness > Fixed::from_int(65) && defender->deterrence_posture > Fixed::from_int(55);
        const bool break_deterrence = unit_dist(rng) < 0.08;
        if (break_deterrence && ((attacker_can_nuclear && attacker_pressure > nuclear_threshold) ||
                                 (defender_can_nuclear && defender_pressure > nuclear_threshold))) {
            result.nuclear_exchange = true;
            const Fixed shock = Fixed::from_double(28.0 + 10.0 * unit_dist(rng));
            attacker->economic_stability = clamp_fixed(attacker->economic_stability - shock, Fixed::from_int(0), Fixed::from_int(100));
            defender->economic_stability = clamp_fixed(defender->economic_stability - shock, Fixed::from_int(0), Fixed::from_int(100));
            attacker->civilian_morale = clamp_fixed(attacker->civilian_morale - shock, Fixed::from_int(0), Fixed::from_int(100));
            defender->civilian_morale = clamp_fixed(defender->civilian_morale - shock, Fixed::from_int(0), Fixed::from_int(100));
            result.infrastructure_damage += Fixed::from_double(20.0);
            result.population_displaced += 12000;
            attacker->nuclear_readiness = clamp_fixed(attacker->nuclear_readiness - Fixed::from_double(12.0), Fixed::from_int(0), Fixed::from_int(100));
            defender->nuclear_readiness = clamp_fixed(defender->nuclear_readiness - Fixed::from_double(12.0), Fixed::from_int(0), Fixed::from_int(100));
            attacker->escalation_level = clamp_fixed(attacker->escalation_level + Fixed::from_int(2), Fixed::from_int(0), Fixed::from_int(5));
            defender->escalation_level = clamp_fixed(defender->escalation_level + Fixed::from_int(2), Fixed::from_int(0), Fixed::from_int(5));
            if (attacker->second_strike_capable || defender->second_strike_capable) {
                attacker->reputation = clamp_fixed(attacker->reputation - Fixed::from_double(12.0), Fixed::from_int(0), Fixed::from_int(100));
                defender->reputation = clamp_fixed(defender->reputation - Fixed::from_double(12.0), Fixed::from_int(0), Fixed::from_int(100));
            }
        }
    }

    int32_t territory_gain = 0;
    if (defender_losses_ratio > attacker_losses_ratio * 1.15) {
        const double momentum = std::max(0.0, defender_losses_ratio - attacker_losses_ratio);
        const uint32_t max_cells = static_cast<uint32_t>(1 + std::floor(momentum * 40.0));
        territory_gain = transfer_border_cells(attacker_id, defender_id, max_cells);
    }

    result.territory_delta = territory_gain;
    if (territory_gain > 0) {
        const Fixed conquest_gain = Fixed::from_double(std::min(8.0, 1.2 + territory_gain * 0.75));
        attacker->resource_oil_reserves = clamp_fixed(attacker->resource_oil_reserves + conquest_gain / Fixed::from_int(4), Fixed::from_int(0), Fixed::from_int(100));
        attacker->resource_minerals_reserves = clamp_fixed(attacker->resource_minerals_reserves + conquest_gain / Fixed::from_int(3), Fixed::from_int(0), Fixed::from_int(100));
        attacker->resource_food_reserves = clamp_fixed(attacker->resource_food_reserves + conquest_gain / Fixed::from_int(4), Fixed::from_int(0), Fixed::from_int(100));
        attacker->resource_rare_earth_reserves = clamp_fixed(attacker->resource_rare_earth_reserves + conquest_gain / Fixed::from_int(5), Fixed::from_int(0), Fixed::from_int(100));
    }
    result.offensive_continues = !result.nuclear_exchange && std::abs(territory_gain) < 3 && defender_losses_ratio > attacker_losses_ratio * 0.95;
    return result;
}

CombatResult World::resolve_offensive_tick(uint16_t attacker_id,
                                           uint16_t defender_id,
                                           uint32_t route_distance,
                                           uint32_t remaining_ticks) {
    const double terrain_roll = 0.92 + 0.16 * std::sin(static_cast<double>(current_tick_) * 0.19 + static_cast<double>(attacker_id));
    const double surprise_roll = 0.90 + 0.22 * std::cos(static_cast<double>(current_tick_) * 0.27 + static_cast<double>(defender_id));
    CombatResult result = resolve_attack(attacker_id,
                                         defender_id,
                                         Fixed::from_double(std::clamp(terrain_roll, 0.65, 1.20)),
                                         Fixed::from_double(std::clamp(surprise_roll, 0.70, 1.25)),
                                         route_distance,
                                         true);
    if (remaining_ticks > 1 && result.offensive_continues) {
        schedule_event(std::make_unique<OffensiveEvent>(attacker_id, defender_id, remaining_ticks - 1, route_distance + 1),
                       current_tick_ + 1);
    }
    return result;
}

void World::resolve_negotiation(uint16_t country_a, uint16_t country_b) {
    Country* a = find_country(country_a);
    Country* b = find_country(country_b);
    if (a == nullptr || b == nullptr) {
        return;
    }

    const Fixed morale_gain = Fixed::from_double(1.50);
    const Fixed econ_gain = Fixed::from_double(1.00);

    a->civilian_morale = clamp_fixed(a->civilian_morale + morale_gain, Fixed::from_int(0), Fixed::from_int(100));
    b->civilian_morale = clamp_fixed(b->civilian_morale + morale_gain, Fixed::from_int(0), Fixed::from_int(100));
    a->economic_stability = clamp_fixed(a->economic_stability + econ_gain, Fixed::from_int(0), Fixed::from_int(100));
    b->economic_stability = clamp_fixed(b->economic_stability + econ_gain, Fixed::from_int(0), Fixed::from_int(100));
    a->resource_reserve = clamp_fixed(a->resource_reserve + Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
    b->resource_reserve = clamp_fixed(b->resource_reserve + Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
    a->intelligence_level = clamp_fixed(a->intelligence_level + Fixed::from_double(0.7), Fixed::from_int(0), Fixed::from_int(100));
    b->intelligence_level = clamp_fixed(b->intelligence_level + Fixed::from_double(0.7), Fixed::from_int(0), Fixed::from_int(100));
    a->supply_level = clamp_fixed(a->supply_level + Fixed::from_double(0.9), Fixed::from_int(0), Fixed::from_int(100));
    b->supply_level = clamp_fixed(b->supply_level + Fixed::from_double(0.9), Fixed::from_int(0), Fixed::from_int(100));
    a->politics.government_stability = clamp_fixed(a->politics.government_stability + Fixed::from_double(0.5), Fixed::from_int(0), Fixed::from_int(100));
    b->politics.government_stability = clamp_fixed(b->politics.government_stability + Fixed::from_double(0.5), Fixed::from_int(0), Fixed::from_int(100));
    a->reputation = clamp_fixed(a->reputation + Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));
    b->reputation = clamp_fixed(b->reputation + Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));

    a->trust_scores[country_b] = clamp_fixed(a->trust_scores[country_b] + Fixed::from_double(3.0), Fixed::from_int(0), Fixed::from_int(100));
    b->trust_scores[country_a] = clamp_fixed(b->trust_scores[country_a] + Fixed::from_double(3.0), Fixed::from_int(0), Fixed::from_int(100));
    if (!is_embargoed_between(*a, *b)) {
        pallas::util::add_unique_id(&a->trade_partners, country_b);
        pallas::util::add_unique_id(&b->trade_partners, country_a);
        a->trade_balance = clamp_fixed(a->trade_balance + Fixed::from_double(0.8), Fixed::from_int(-100), Fixed::from_int(100));
        b->trade_balance = clamp_fixed(b->trade_balance + Fixed::from_double(0.8), Fixed::from_int(-100), Fixed::from_int(100));
    }

    if (a->diplomatic_stance == DiplomaticStance::Aggressive) {
        a->diplomatic_stance = DiplomaticStance::Neutral;
    }
    if (b->diplomatic_stance == DiplomaticStance::Aggressive) {
        b->diplomatic_stance = DiplomaticStance::Neutral;
    }
}

bool World::ScheduledEventCompare::operator()(const ScheduledEvent& lhs, const ScheduledEvent& rhs) const {
    if (lhs.tick == rhs.tick) {
        return lhs.sequence > rhs.sequence;
    }
    return lhs.tick > rhs.tick;
}

Country* World::find_country(uint16_t id) {
    const auto it = country_index_by_id_.find(id);
    if (it != country_index_by_id_.end() && it->second < countries_.size()) {
        return &countries_[it->second];
    }
    for (size_t i = 0; i < countries_.size(); ++i) {
        if (countries_[i].id == id) {
            country_index_by_id_[id] = i;
            return &countries_[i];
        }
    }
    return nullptr;
}

const Country* World::find_country(uint16_t id) const {
    const auto it = country_index_by_id_.find(id);
    if (it != country_index_by_id_.end() && it->second < countries_.size()) {
        return &countries_[it->second];
    }
    for (const Country& c : countries_) {
        if (c.id == id) {
            return &c;
        }
    }
    return nullptr;
}

void World::rebuild_country_index() {
    country_index_by_id_.clear();
    country_index_by_id_.reserve(countries_.size());
    for (size_t i = 0; i < countries_.size(); ++i) {
        country_index_by_id_[countries_[i].id] = i;
    }
}

void World::recompute_territory_cells() {
    for (Country& c : countries_) {
        c.territory_cells = 0;
    }
    for (uint16_t id : map_.flattened_country_ids()) {
        Country* c = find_country(id);
        if (c != nullptr) {
            ++c->territory_cells;
        }
    }
}

int32_t World::transfer_border_cells(uint16_t attacker_id, uint16_t defender_id, uint32_t max_cells) {
    if (map_.width() == 0 || map_.height() == 0 || max_cells == 0) {
        return 0;
    }

    uint32_t changed = 0;
    for (uint32_t y = 0; y < map_.height() && changed < max_cells; ++y) {
        for (uint32_t x = 0; x < map_.width() && changed < max_cells; ++x) {
            if (map_.at(x, y) != defender_id) {
                continue;
            }
            bool has_attacker_neighbor = false;
            if (x > 0 && map_.at(x - 1, y) == attacker_id) {
                has_attacker_neighbor = true;
            }
            if (!has_attacker_neighbor && x + 1 < map_.width() && map_.at(x + 1, y) == attacker_id) {
                has_attacker_neighbor = true;
            }
            if (!has_attacker_neighbor && y > 0 && map_.at(x, y - 1) == attacker_id) {
                has_attacker_neighbor = true;
            }
            if (!has_attacker_neighbor && y + 1 < map_.height() && map_.at(x, y + 1) == attacker_id) {
                has_attacker_neighbor = true;
            }
            if (!has_attacker_neighbor) {
                continue;
            }
            map_.set(x, y, attacker_id);
            ++changed;
        }
    }
    if (changed > 0) {
        recompute_territory_cells();
    }
    return static_cast<int32_t>(changed);
}

uint64_t World::derive_event_seed(uint16_t attacker_id, uint16_t defender_id) const {
    uint64_t value = base_seed_;
    value ^= mix_u64(current_tick_ + 0x9e3779b97f4a7c15ULL);
    value ^= mix_u64(static_cast<uint64_t>(attacker_id) << 16U | static_cast<uint64_t>(defender_id));
    value ^= mix_u64(next_event_sequence_ + static_cast<uint64_t>(random_seed_log_.size()) + 17ULL);
    return mix_u64(value);
}

bool World::has_active_trade_link(const Country& a, uint16_t other_id) const {
    return pallas::util::contains_id(a.trade_partners, other_id);
}

bool World::is_embargoed_between(const Country& a, const Country& b) const {
        return pallas::util::contains_id(a.embargoed_country_ids, b.id) ||
            pallas::util::contains_id(b.embargoed_country_ids, a.id);
}

void World::update_trade_and_internal_politics() {
    if (countries_.empty()) {
        return;
    }

    rebuild_country_index();

    for (Country& country : countries_) {
        auto prune = [&](std::vector<uint16_t>* values, std::unordered_map<uint16_t, uint64_t>* expiry) {
            if (values == nullptr || expiry == nullptr) {
                return;
            }
            values->erase(std::remove_if(values->begin(), values->end(), [&](uint16_t other_id) {
                const auto it = expiry->find(other_id);
                if (it == expiry->end()) {
                    return false;
                }
                return it->second <= current_tick_;
            }), values->end());
            for (auto it = expiry->begin(); it != expiry->end();) {
                if (it->second <= current_tick_) {
                    it = expiry->erase(it);
                } else {
                    ++it;
                }
            }
        };
        prune(&country.has_defense_pact_with, &country.defense_pact_expiry_ticks);
        prune(&country.has_non_aggression_with, &country.non_aggression_expiry_ticks);
        prune(&country.has_trade_treaty_with, &country.trade_treaty_expiry_ticks);
    }

    std::vector<Fixed> next_trade_balance(countries_.size(), Fixed::from_int(0));
    const double global_shock_roll = normalized_event_roll(base_seed_, current_tick_ + 0x5f3759dfULL);
    const bool global_recession = global_shock_roll < 0.05;
    const bool shipping_crisis = global_shock_roll > 0.78 && global_shock_roll < 0.84;

    for (size_t i = 0; i < countries_.size(); ++i) {
        Country& country = countries_[i];
        for (uint16_t embargoed : country.embargoed_country_ids) {
            const auto it = country_index_by_id_.find(embargoed);
            if (it == country_index_by_id_.end() || it->second >= countries_.size()) {
                continue;
            }
            Country& other = countries_[it->second];
            pallas::util::erase_id(&country.trade_partners, other.id);
            pallas::util::erase_id(&other.trade_partners, country.id);
            country.economic_stability = clamp_fixed(country.economic_stability - Fixed::from_double(0.45), Fixed::from_int(0), Fixed::from_int(100));
            other.economic_stability = clamp_fixed(other.economic_stability - Fixed::from_double(0.30), Fixed::from_int(0), Fixed::from_int(100));
            country.trust_scores[other.id] = clamp_fixed(country.trust_scores[other.id] - Fixed::from_double(0.9), Fixed::from_int(0), Fixed::from_int(100));
        }
    }

    for (size_t i = 0; i < countries_.size(); ++i) {
        Country& a = countries_[i];
        for (uint16_t partner_id : a.trade_partners) {
            const auto it = country_index_by_id_.find(partner_id);
            if (it == country_index_by_id_.end() || it->second <= i || it->second >= countries_.size()) {
                continue;
            }
            Country& b = countries_[it->second];
            if (!has_active_trade_link(b, a.id) || is_embargoed_between(a, b)) {
                continue;
            }

            const double a_supply = a.resource_food_reserves.to_double() + a.resource_oil_reserves.to_double() * 0.7 +
                a.resource_minerals_reserves.to_double() * 0.6 + a.resource_rare_earth_reserves.to_double() * 0.8;
            const double b_supply = b.resource_food_reserves.to_double() + b.resource_oil_reserves.to_double() * 0.7 +
                b.resource_minerals_reserves.to_double() * 0.6 + b.resource_rare_earth_reserves.to_double() * 0.8;
            const double a_demand = a.industrial_output.to_double() * 0.9 + a.military_upkeep.to_double() * 0.8 + a.draft_level.to_double() * 0.3;
            const double b_demand = b.industrial_output.to_double() * 0.9 + b.military_upkeep.to_double() * 0.8 + b.draft_level.to_double() * 0.3;
                const double treaty_bonus = (pallas::util::contains_id(a.has_trade_treaty_with, b.id) &&
                                                      pallas::util::contains_id(b.has_trade_treaty_with, a.id))
                                                          ? 1.18
                                                          : 1.0;
            const double trust_factor = (0.65 + std::min(0.35, (a.trust_scores[b.id].to_double() + b.trust_scores[a.id].to_double()) / 260.0)) * treaty_bonus;
            const double gross_flow = std::clamp(((a_supply - a_demand) - (b_supply - b_demand)) / 42.0, -6.0, 6.0) * trust_factor;

            next_trade_balance[i] += Fixed::from_double(gross_flow);
            next_trade_balance[it->second] -= Fixed::from_double(gross_flow);

            const Fixed econ_bonus = Fixed::from_double(std::max(0.2, 0.5 + std::abs(gross_flow) * 0.18));
            a.economic_stability = clamp_fixed(a.economic_stability + econ_bonus, Fixed::from_int(0), Fixed::from_int(100));
            b.economic_stability = clamp_fixed(b.economic_stability + econ_bonus, Fixed::from_int(0), Fixed::from_int(100));
            a.civilian_morale = clamp_fixed(a.civilian_morale + econ_bonus / Fixed::from_int(3), Fixed::from_int(0), Fixed::from_int(100));
            b.civilian_morale = clamp_fixed(b.civilian_morale + econ_bonus / Fixed::from_int(3), Fixed::from_int(0), Fixed::from_int(100));
        }
    }

    for (size_t i = 0; i < countries_.size(); ++i) {
        Country& c = countries_[i];

        const double industry = c.industrial_output.to_double();
        const double upkeep = c.military_upkeep.to_double();
        const double draft = c.draft_level.to_double();
        const double extraction_efficiency = 0.25 + c.resources.oil.to_double() / 300.0 + c.resources.minerals.to_double() / 400.0;

        const Fixed oil_drain = Fixed::from_double(0.18 + industry * 0.010 + upkeep * 0.007 + draft * 0.004);
        const Fixed mineral_drain = Fixed::from_double(0.16 + industry * 0.009 + upkeep * 0.004);
        const Fixed food_drain = Fixed::from_double(0.14 + industry * 0.005 + draft * 0.006 + c.population / 40000000.0);
        const Fixed rare_drain = Fixed::from_double(0.08 + c.technology_level.to_double() * 0.006 + c.military.units_air_bomber.to_double() / 260.0 + c.military.units_naval_submarine.to_double() / 220.0);

        c.resource_oil_reserves = clamp_fixed(c.resource_oil_reserves - oil_drain, Fixed::from_int(0), Fixed::from_int(100));
        c.resource_minerals_reserves = clamp_fixed(c.resource_minerals_reserves - mineral_drain, Fixed::from_int(0), Fixed::from_int(100));
        c.resource_food_reserves = clamp_fixed(c.resource_food_reserves - food_drain, Fixed::from_int(0), Fixed::from_int(100));
        c.resource_rare_earth_reserves = clamp_fixed(c.resource_rare_earth_reserves - rare_drain, Fixed::from_int(0), Fixed::from_int(100));

        if (global_recession) {
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(1.8 + c.trade_partners.size() * 0.15), Fixed::from_int(0), Fixed::from_int(100));
            c.civilian_morale = clamp_fixed(c.civilian_morale - Fixed::from_double(0.9), Fixed::from_int(0), Fixed::from_int(100));
        }
        if (shipping_crisis && !c.trade_partners.empty()) {
            c.logistics_capacity = clamp_fixed(c.logistics_capacity - Fixed::from_double(1.1 + c.trade_partners.size() * 0.1), Fixed::from_int(0), Fixed::from_int(100));
            c.trade_balance = clamp_fixed(c.trade_balance - Fixed::from_double(2.5), Fixed::from_int(-100), Fixed::from_int(100));
        }

        c.trade_balance = clamp_fixed(next_trade_balance[i], Fixed::from_int(-100), Fixed::from_int(100));

        const Fixed reserve_average = (c.resource_oil_reserves + c.resource_minerals_reserves + c.resource_food_reserves + c.resource_rare_earth_reserves) / Fixed::from_int(4);
        c.resource_reserve = reserve_average;
        c.supply_capacity = clamp_fixed(Fixed::from_double(c.logistics_capacity.to_double() * 0.72 + c.industrial_output.to_double() * 0.18 + c.resource_reserve.to_double() * 0.10),
                        Fixed::from_int(0),
                        Fixed::from_int(100));
        c.supply_level = clamp_fixed(Fixed::from_double(c.supply_level.to_double() * 0.78 + c.supply_capacity.to_double() * 0.22 - c.war_weariness.to_double() * 0.02 - c.adjacent_country_ids.size() * 0.15),
                         Fixed::from_int(0),
                         Fixed::from_int(100));

        c.military_upkeep = clamp_fixed(
            Fixed::from_double(c.military.weighted_total().to_double() / 180.0 + c.draft_level.to_double() * 0.22),
            Fixed::from_int(0),
            Fixed::from_int(100));

        Fixed military_shift = Fixed::from_double((upkeep + draft * 0.6 + c.war_duration_ticks * 0.12) / 22.0);
        Fixed industrial_shift = Fixed::from_double((industry + c.trade_partners.size() * 6.0 + extraction_efficiency * 30.0) / 28.0);
        Fixed civilian_shift = Fixed::from_double((c.civilian_morale.to_double() + c.economic_stability.to_double() - c.war_weariness.to_double()) / 45.0);

        c.faction_military = clamp_fixed(c.faction_military + military_shift - Fixed::from_double(c.trade_partners.size() * 0.15), Fixed::from_int(0), Fixed::from_int(100));
        c.faction_industrial = clamp_fixed(c.faction_industrial + industrial_shift - Fixed::from_double(c.politics.corruption.to_double() / 90.0), Fixed::from_int(0), Fixed::from_int(100));
        c.faction_civilian = clamp_fixed(c.faction_civilian + civilian_shift - Fixed::from_double(c.draft_level.to_double() / 80.0), Fixed::from_int(0), Fixed::from_int(100));

        const double faction_sum = std::max(1.0,
            c.faction_military.to_double() + c.faction_industrial.to_double() + c.faction_civilian.to_double());
        c.faction_military = Fixed::from_double(c.faction_military.to_double() * 100.0 / faction_sum);
        c.faction_industrial = Fixed::from_double(c.faction_industrial.to_double() * 100.0 / faction_sum);
        c.faction_civilian = Fixed::from_double(c.faction_civilian.to_double() * 100.0 / faction_sum);

        const double weariness_delta = c.recent_combat_losses.to_double() * 0.06 + c.war_duration_ticks * 0.03 + draft * 0.02 -
            c.trade_partners.size() * 0.08 - c.civilian_morale.to_double() * 0.006;
        c.war_weariness = clamp_fixed(c.war_weariness + Fixed::from_double(weariness_delta), Fixed::from_int(0), Fixed::from_int(100));

        if (c.recent_combat_losses > Fixed::from_double(0.4)) {
            c.recent_combat_losses = clamp_fixed(c.recent_combat_losses - Fixed::from_double(0.4), Fixed::from_int(0), Fixed::from_int(100));
        } else {
            c.recent_combat_losses = Fixed::from_int(0);
            if (c.war_duration_ticks > 0) {
                --c.war_duration_ticks;
            }
        }

        if (c.war_weariness > Fixed::from_int(55)) {
            c.civilian_morale = clamp_fixed(c.civilian_morale - (c.war_weariness / Fixed::from_int(70)), Fixed::from_int(0), Fixed::from_int(100));
            c.politics.public_dissent = clamp_fixed(c.politics.public_dissent + (c.war_weariness / Fixed::from_int(90)), Fixed::from_int(0), Fixed::from_int(100));
        }

        const double coup_score = c.faction_military.to_double() * 0.35 + c.politics.public_dissent.to_double() * 0.28 +
            (100.0 - c.politics.government_stability.to_double()) * 0.24 + (100.0 - c.economic_stability.to_double()) * 0.16 -
            c.faction_civilian.to_double() * 0.18;
        c.coup_risk = clamp_fixed(Fixed::from_double(coup_score / 0.9), Fixed::from_int(0), Fixed::from_int(100));

        c.election_cycle -= 1;
        if (c.election_cycle <= 0) {
            const double performance = c.civilian_morale.to_double() + c.economic_stability.to_double() + c.trade_balance.to_double() * 0.4 - c.war_weariness.to_double();
            if (performance < 85.0 && c.faction_military > c.faction_civilian) {
                c.diplomatic_stance = DiplomaticStance::Aggressive;
                c.politics.government_stability = clamp_fixed(c.politics.government_stability - Fixed::from_double(2.5), Fixed::from_int(0), Fixed::from_int(100));
                c.faction_military = clamp_fixed(c.faction_military + Fixed::from_double(4.0), Fixed::from_int(0), Fixed::from_int(100));
            } else if (performance > 150.0 || c.faction_civilian >= c.faction_military) {
                c.diplomatic_stance = DiplomaticStance::Pacifist;
                c.politics.government_stability = clamp_fixed(c.politics.government_stability + Fixed::from_double(4.0), Fixed::from_int(0), Fixed::from_int(100));
                c.faction_civilian = clamp_fixed(c.faction_civilian + Fixed::from_double(3.5), Fixed::from_int(0), Fixed::from_int(100));
            } else {
                c.diplomatic_stance = DiplomaticStance::Neutral;
                c.politics.government_stability = clamp_fixed(c.politics.government_stability + Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
                c.faction_industrial = clamp_fixed(c.faction_industrial + Fixed::from_double(2.0), Fixed::from_int(0), Fixed::from_int(100));
            }
            c.election_cycle = 10 + static_cast<int>((mix_u64(base_seed_ + c.id + current_tick_) % 9ULL));
            c.politics.public_dissent = clamp_fixed(c.politics.public_dissent - Fixed::from_double(3.0), Fixed::from_int(0), Fixed::from_int(100));
        }

        const double coup_roll = normalized_event_roll(base_seed_ + c.id, current_tick_ + c.id * 17ULL);
        if (c.coup_risk > Fixed::from_int(72) && c.faction_military > Fixed::from_int(42) && coup_roll < c.coup_risk.to_double() / 180.0) {
            c.diplomatic_stance = DiplomaticStance::Aggressive;
            c.politics.government_stability = clamp_fixed(Fixed::from_double(52.0 - coup_roll * 10.0), Fixed::from_int(0), Fixed::from_int(100));
            c.civilian_morale = clamp_fixed(c.civilian_morale - Fixed::from_double(9.0), Fixed::from_int(0), Fixed::from_int(100));
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(6.0), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_military = clamp_fixed(c.faction_military + Fixed::from_double(10.0), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_civilian = clamp_fixed(c.faction_civilian - Fixed::from_double(8.0), Fixed::from_int(0), Fixed::from_int(100));
            c.trade_partners.clear();
            c.embargoed_country_ids.clear();
            c.election_cycle = 18;
        }

        if (c.resource_oil_reserves < Fixed::from_int(12) || c.resource_food_reserves < Fixed::from_int(12)) {
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(1.4), Fixed::from_int(0), Fixed::from_int(100));
            c.civilian_morale = clamp_fixed(c.civilian_morale - Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
            c.politics.public_dissent = clamp_fixed(c.politics.public_dissent + Fixed::from_double(1.1), Fixed::from_int(0), Fixed::from_int(100));
        }

        for (uint16_t neighbor_id : c.adjacent_country_ids) {
            const Country* neighbor = find_country(neighbor_id);
            if (neighbor == nullptr) {
                continue;
            }
            const double passive_intel = c.intelligence_level.to_double() * 0.14 + c.technology.cyber_warfare.to_double() * 0.05 - neighbor->technology.electronic_warfare.to_double() * 0.03;
            c.intel_on_enemy[neighbor_id] = clamp_fixed(Fixed::from_double(c.intel_on_enemy[neighbor_id].to_double() * 0.86 + passive_intel), Fixed::from_int(0), Fixed::from_int(100));
        }
    }
}

void World::prune_recent_betrayal_memory(Country& country) {
    constexpr uint64_t kBetrayalWindowTicks = 12;
    country.betrayal_tick_log.erase(
        std::remove_if(country.betrayal_tick_log.begin(),
                       country.betrayal_tick_log.end(),
                       [&](uint64_t tick) {
                           return current_tick_ > tick && (current_tick_ - tick) > kBetrayalWindowTicks;
                       }),
        country.betrayal_tick_log.end());
}

void World::update_opponent_models() {
    if (countries_.empty()) {
        return;
    }

    rebuild_country_index();

    for (Country& observer : countries_) {
        prune_recent_betrayal_memory(observer);

        double mean_confidence = 0.0;
        size_t confidence_count = 0;
        double threat_pressure = 0.0;
        const double own_strength = std::max(1.0, country_strength_estimate(observer).to_double());

        for (uint16_t neighbor_id : observer.adjacent_country_ids) {
            const Country* other = find_country(neighbor_id);
            if (other == nullptr) {
                continue;
            }

            const double actual_strength = std::max(1.0, country_strength_estimate(*other).to_double());
            const auto intel_it = observer.intel_on_enemy.find(neighbor_id);
            const double direct_intel = intel_it == observer.intel_on_enemy.end() ? 0.0 : intel_it->second.to_double();
            const double passive_intel = observer.intelligence_level.to_double() * 0.55;
            const double visibility = std::clamp((direct_intel * 0.72 + passive_intel * 0.28) / 100.0, 0.0, 1.0);

            const auto trust_it = observer.trust_scores.find(neighbor_id);
            const double trust = trust_it == observer.trust_scores.end() ? 50.0 : trust_it->second.to_double();
            const double suspicion_bias = 1.0 + std::max(0.0, 55.0 - trust) / 180.0;

            const auto prior_belief_it = observer.believed_army_size.find(neighbor_id);
            const double prior_belief = prior_belief_it == observer.believed_army_size.end()
                ? actual_strength * (0.68 + (1.0 - visibility) * 0.25)
                : std::max(1.0, prior_belief_it->second.to_double());
            const double observed_strength = actual_strength * (0.80 + visibility * 0.20) * suspicion_bias;
            const double blend = 0.14 + visibility * 0.56;
            observer.believed_army_size[neighbor_id] = Fixed::from_double(
                std::max(0.0, prior_belief * (1.0 - blend) + observed_strength * blend));

            const auto prior_conf_it = observer.opponent_model_confidence.find(neighbor_id);
            const double prior_conf = prior_conf_it == observer.opponent_model_confidence.end()
                ? 28.0
                : prior_conf_it->second.to_double();
            const double betrayal_bonus = std::min(12.0, static_cast<double>(observer.betrayal_tick_log.size()) * 3.0);
            const double updated_conf = std::clamp(prior_conf * 0.74 + (18.0 + visibility * 78.0 + betrayal_bonus) * 0.26,
                                                   5.0,
                                                   100.0);
            observer.opponent_model_confidence[neighbor_id] = Fixed::from_double(updated_conf);

            mean_confidence += updated_conf;
            ++confidence_count;
            threat_pressure += std::max(0.0, actual_strength - own_strength) / own_strength;
        }

        const double avg_confidence = confidence_count == 0 ? 25.0 : mean_confidence / static_cast<double>(confidence_count);
        const double betrayal_pressure = static_cast<double>(observer.betrayal_tick_log.size()) * 0.35;
        const double ally_buffer = std::min(0.6, static_cast<double>(observer.allied_country_ids.size()) * 0.08);
        const double planning_depth = std::clamp(1.0 + avg_confidence / 38.0 + betrayal_pressure + threat_pressure * 0.65 + ally_buffer,
                                                 1.0,
                                                 6.0);
        observer.strategic_depth = Fixed::from_double(planning_depth);
    }
}

void World::update_country_dynamics_parallel() {
    if (countries_.empty()) {
        return;
    }

    rebuild_country_index();

    std::vector<Fixed> next_morale(countries_.size());
    std::vector<Fixed> next_econ(countries_.size());
    std::vector<Fixed> next_infantry(countries_.size());
    std::vector<Fixed> next_armor(countries_.size());
    std::vector<Fixed> next_artillery(countries_.size());
    std::vector<Fixed> next_fighters(countries_.size());
    std::vector<Fixed> next_bombers(countries_.size());
    std::vector<Fixed> next_surface(countries_.size());
    std::vector<Fixed> next_submarines(countries_.size());
    std::vector<Fixed> next_logistics(countries_.size());
    std::vector<Fixed> next_intel(countries_.size());
    std::vector<Fixed> next_industry(countries_.size());
    std::vector<Fixed> next_technology(countries_.size());
    std::vector<Fixed> next_reserve(countries_.size());
    std::vector<Fixed> next_supply(countries_.size());
    std::vector<Fixed> next_supply_capacity(countries_.size());
    std::vector<Fixed> next_weather(countries_.size());
    std::vector<Fixed> next_season(countries_.size());
    std::vector<Fixed> next_gov(countries_.size());
    std::vector<Fixed> next_dissent(countries_.size());
    std::vector<Fixed> next_missile_def(countries_.size());
    std::vector<Fixed> next_cyber(countries_.size());
    std::vector<Fixed> next_draft(countries_.size());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(countries_.size()); ++i) {
        const Country& country = countries_[static_cast<size_t>(i)];

        Fixed border_pressure = Fixed::from_int(0);
        for (uint16_t neighbor_id : country.adjacent_country_ids) {
            const auto it = country_index_by_id_.find(neighbor_id);
            if (it == country_index_by_id_.end() || it->second >= countries_.size()) {
                continue;
            }
            const Country& neighbor = countries_[it->second];
            bool allied = false;
            for (uint16_t ally_id : country.allied_country_ids) {
                if (ally_id == neighbor_id) {
                    allied = true;
                    break;
                }
            }
            if (allied) {
                continue;
            }
            const Fixed diff = neighbor.military.weighted_total() - country.military.weighted_total();
            if (diff > Fixed::from_int(0)) {
                border_pressure += diff / Fixed::from_int(200);
            }
        }

        Fixed stance_morale_modifier = Fixed::from_double(0.05);
        Fixed stance_econ_modifier = Fixed::from_double(0.03);
        if (country.diplomatic_stance == DiplomaticStance::Aggressive) {
            stance_morale_modifier = Fixed::from_double(-0.10);
            stance_econ_modifier = Fixed::from_double(-0.12);
        } else if (country.diplomatic_stance == DiplomaticStance::Pacifist) {
            stance_morale_modifier = Fixed::from_double(0.12);
            stance_econ_modifier = Fixed::from_double(0.08);
        }

        const Fixed morale_decay = border_pressure / Fixed::from_int(20);
        const Fixed econ_decay = border_pressure / Fixed::from_int(25);
        const Fixed industry_boost = country.industrial_output / Fixed::from_int(140);
        const Fixed logistics_boost = country.logistics_capacity / Fixed::from_int(180);
        const Fixed reserve_boost = country.resource_reserve / Fixed::from_int(220);
        const Fixed politics_drag = country.politics.public_dissent / Fixed::from_int(140);
        const Fixed upkeep_drag = country.military_upkeep / Fixed::from_int(170);
        const Fixed weariness_drag = country.war_weariness / Fixed::from_int(150);
        const Fixed trust_bonus = country.adjacent_country_ids.empty()
            ? Fixed::from_double(0.0)
            : Fixed::from_double([&]() {
                double total = 0.0;
                for (uint16_t neighbor_id : country.adjacent_country_ids) {
                    const auto it = country.trust_scores.find(neighbor_id);
                    if (it != country.trust_scores.end()) {
                        total += it->second.to_double();
                    }
                }
                return (total / std::max<size_t>(1, country.adjacent_country_ids.size())) / 350.0;
            }());
        const Fixed infantry_regen = clamp_fixed((country.economic_stability / Fixed::from_int(120)) + industry_boost + logistics_boost + reserve_boost,
                             Fixed::from_int(0), Fixed::from_double(1.8));
        const Fixed armor_regen = clamp_fixed((country.industrial_output / Fixed::from_int(180)) + (country.resources.oil / Fixed::from_int(260)) - weariness_drag,
                              Fixed::from_int(0), Fixed::from_double(0.9));
        const Fixed artillery_regen = clamp_fixed((country.industrial_output / Fixed::from_int(200)) + (country.resources.minerals / Fixed::from_int(280)) - (border_pressure / Fixed::from_int(40)),
                              Fixed::from_int(0), Fixed::from_double(0.8));
        const Fixed fighter_regen = clamp_fixed((country.technology_level / Fixed::from_int(220)) + (country.logistics_capacity / Fixed::from_int(260)) - (country.weather_severity / Fixed::from_int(320)),
                            Fixed::from_int(0), Fixed::from_double(0.55));
        const Fixed bomber_regen = clamp_fixed((country.technology.drone_operations / Fixed::from_int(240)) + (country.resources.rare_earth / Fixed::from_int(320)) - weariness_drag,
                               Fixed::from_int(0), Fixed::from_double(0.35));
        const Fixed surface_regen = clamp_fixed((country.resources.oil / Fixed::from_int(320)) + (country.industrial_output / Fixed::from_int(300)) - (border_pressure / Fixed::from_int(55)),
                            Fixed::from_int(0), Fixed::from_double(0.28));
        const Fixed submarine_regen = clamp_fixed((country.intelligence_level / Fixed::from_int(320)) + (country.resources.rare_earth / Fixed::from_int(360)) - (border_pressure / Fixed::from_int(60)),
                              Fixed::from_int(0), Fixed::from_double(0.24));

        next_morale[static_cast<size_t>(i)] = clamp_fixed(country.civilian_morale + stance_morale_modifier - morale_decay - weariness_drag, Fixed::from_int(0), Fixed::from_int(100));
        next_econ[static_cast<size_t>(i)] = clamp_fixed(country.economic_stability + stance_econ_modifier + trust_bonus - econ_decay - politics_drag - upkeep_drag, Fixed::from_int(0), Fixed::from_int(100));
        next_infantry[static_cast<size_t>(i)] = clamp_fixed(country.military.units_infantry + infantry_regen - (border_pressure / Fixed::from_int(10)) - (country.weather_severity / Fixed::from_int(280)) + (country.draft_level / Fixed::from_int(180)), Fixed::from_int(0), Fixed::from_int(1000000));
        next_armor[static_cast<size_t>(i)] = clamp_fixed(country.military.units_armor + armor_regen - (border_pressure / Fixed::from_int(18)) - (country.weather_severity / Fixed::from_int(360)), Fixed::from_int(0), Fixed::from_int(1000000));
        next_artillery[static_cast<size_t>(i)] = clamp_fixed(country.military.units_artillery + artillery_regen - (border_pressure / Fixed::from_int(16)), Fixed::from_int(0), Fixed::from_int(1000000));
        next_fighters[static_cast<size_t>(i)] = clamp_fixed(country.military.units_air_fighter + fighter_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_bombers[static_cast<size_t>(i)] = clamp_fixed(country.military.units_air_bomber + bomber_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_surface[static_cast<size_t>(i)] = clamp_fixed(country.military.units_naval_surface + surface_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_submarines[static_cast<size_t>(i)] = clamp_fixed(country.military.units_naval_submarine + submarine_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_logistics[static_cast<size_t>(i)] = clamp_fixed(country.logistics_capacity + (country.economic_stability / Fixed::from_int(260)) - (border_pressure / Fixed::from_int(30)) - weariness_drag,
                                                             Fixed::from_int(0), Fixed::from_int(100));
        next_intel[static_cast<size_t>(i)] = clamp_fixed(country.intelligence_level + (country.technology_level / Fixed::from_int(300)) - (border_pressure / Fixed::from_int(35)),
                                                         Fixed::from_int(0), Fixed::from_int(100));
        next_industry[static_cast<size_t>(i)] = clamp_fixed(country.industrial_output + (country.economic_stability / Fixed::from_int(240)) - (border_pressure / Fixed::from_int(28)) - upkeep_drag,
                                                            Fixed::from_int(0), Fixed::from_int(100));
        next_technology[static_cast<size_t>(i)] = clamp_fixed(country.technology_level + (country.intelligence_level / Fixed::from_int(320)) - (border_pressure / Fixed::from_int(40)),
                                                              Fixed::from_int(0), Fixed::from_int(100));
        next_reserve[static_cast<size_t>(i)] = clamp_fixed(country.resource_reserve + (country.industrial_output / Fixed::from_int(260)) - (border_pressure / Fixed::from_int(22)),
                                                           Fixed::from_int(0), Fixed::from_int(100));
        next_supply[static_cast<size_t>(i)] = clamp_fixed(country.supply_level + (country.resource_reserve / Fixed::from_int(250)) + (country.resources.food / Fixed::from_int(320)) - (border_pressure / Fixed::from_int(25)),
                                  Fixed::from_int(0), Fixed::from_int(100));
        next_supply_capacity[static_cast<size_t>(i)] = clamp_fixed(country.supply_capacity + (country.logistics_capacity / Fixed::from_int(260)) + (country.industrial_output / Fixed::from_int(300)) - (border_pressure / Fixed::from_int(45)),
                                       Fixed::from_int(0), Fixed::from_int(100));
        next_weather[static_cast<size_t>(i)] = clamp_fixed(country.weather_severity + Fixed::from_double(std::sin((static_cast<double>(current_tick_) + i) * 0.03) * 0.8),
                                   Fixed::from_int(0), Fixed::from_int(100));
        next_season[static_cast<size_t>(i)] = clamp_fixed(country.seasonal_effect + Fixed::from_double(std::cos((static_cast<double>(current_tick_) + i * 7) * 0.02) * 1.1),
                                  Fixed::from_int(0), Fixed::from_int(100));
        next_gov[static_cast<size_t>(i)] = clamp_fixed(country.politics.government_stability + (country.economic_stability / Fixed::from_int(330)) - (country.politics.public_dissent / Fixed::from_int(210)),
                                   Fixed::from_int(0), Fixed::from_int(100));
        next_dissent[static_cast<size_t>(i)] = clamp_fixed(country.politics.public_dissent + (border_pressure / Fixed::from_int(20)) - (country.civilian_morale / Fixed::from_int(260)) + weariness_drag,
                                   Fixed::from_int(0), Fixed::from_int(100));
        next_missile_def[static_cast<size_t>(i)] = clamp_fixed(country.technology.missile_defense + (country.technology_level / Fixed::from_int(500)),
                                       Fixed::from_int(0), Fixed::from_int(100));
        next_cyber[static_cast<size_t>(i)] = clamp_fixed(country.technology.cyber_warfare + (country.intelligence_level / Fixed::from_int(450)),
                                 Fixed::from_int(0), Fixed::from_int(100));
        next_draft[static_cast<size_t>(i)] = clamp_fixed(country.draft_level + (border_pressure / Fixed::from_int(35)) - (country.war_weariness / Fixed::from_int(180)),
                     Fixed::from_int(0), Fixed::from_int(100));
    }

    for (size_t i = 0; i < countries_.size(); ++i) {
        countries_[i].civilian_morale = next_morale[i];
        countries_[i].economic_stability = next_econ[i];
        countries_[i].military.units_infantry = next_infantry[i];
        countries_[i].military.units_armor = next_armor[i];
        countries_[i].military.units_artillery = next_artillery[i];
        countries_[i].military.units_air_fighter = next_fighters[i];
        countries_[i].military.units_air_bomber = next_bombers[i];
        countries_[i].military.units_naval_surface = next_surface[i];
        countries_[i].military.units_naval_submarine = next_submarines[i];
        countries_[i].logistics_capacity = next_logistics[i];
        countries_[i].intelligence_level = next_intel[i];
        countries_[i].industrial_output = next_industry[i];
        countries_[i].technology_level = next_technology[i];
        countries_[i].resource_reserve = next_reserve[i];
        countries_[i].supply_level = next_supply[i];
        countries_[i].supply_capacity = next_supply_capacity[i];
        countries_[i].supply_stockpile = next_supply[i];
        countries_[i].weather_severity = next_weather[i];
        countries_[i].seasonal_effect = next_season[i];
        countries_[i].politics.government_stability = next_gov[i];
        countries_[i].politics.public_dissent = next_dissent[i];
        countries_[i].technology.missile_defense = next_missile_def[i];
        countries_[i].technology.cyber_warfare = next_cyber[i];
        countries_[i].draft_level = next_draft[i];
    }
}

}  // namespace sim