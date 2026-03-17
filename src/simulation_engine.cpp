#include "simulation_engine.h"
#include "common_utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <deque>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <omp.h>

namespace sim {

namespace {

constexpr uint32_t kMapMagic = 0x50414C4D;
constexpr uint32_t kMapVersion = 2;

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

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double deterministic_roll01(uint64_t seed, uint64_t salt) {
    const uint64_t mixed = mix_u64(seed ^ mix_u64(salt + 0x9e3779b97f4a7c15ULL));
    return static_cast<double>(mixed % 100000ULL) / 100000.0;
}

int dominant_faction_index(const Country& country) {
    const std::array<double, 4> influence = {
        country.faction_hawks.to_double(),
        country.faction_economic_liberals.to_double(),
        country.faction_nationalists.to_double(),
        country.faction_populists.to_double(),
    };
    size_t best = 0;
    for (size_t i = 1; i < influence.size(); ++i) {
        if (influence[i] > influence[best]) {
            best = i;
        }
    }
    return static_cast<int>(best);
}

void reroll_leader_traits(Country* country, uint64_t base_seed, uint64_t salt, int preferred_faction) {
    if (country == nullptr) {
        return;
    }

    const double r1 = deterministic_roll01(base_seed, salt + 11ULL);
    const double r2 = deterministic_roll01(base_seed, salt + 23ULL);
    const double r3 = deterministic_roll01(base_seed, salt + 47ULL);
    const double r4 = deterministic_roll01(base_seed, salt + 71ULL);

    double aggressive = 25.0 + r1 * 60.0;
    double diplomatic = 25.0 + r2 * 60.0;
    double corrupt = 12.0 + r3 * 55.0;
    double competent = 20.0 + r4 * 65.0;

    if (country->regime_type == RegimeType::Democratic) {
        diplomatic += 10.0;
        corrupt -= 6.0;
    } else if (country->regime_type == RegimeType::Authoritarian) {
        aggressive += 9.0;
        corrupt += 8.0;
        diplomatic -= 8.0;
    }

    if (preferred_faction == 0) {
        aggressive += 12.0;
        diplomatic -= 6.0;
    } else if (preferred_faction == 1) {
        diplomatic += 8.0;
        competent += 6.0;
    } else if (preferred_faction == 2) {
        aggressive += 7.0;
        diplomatic -= 4.0;
    } else if (preferred_faction == 3) {
        corrupt += 6.0;
    }

    country->leader_traits.aggressive = clamp_fixed(Fixed::from_double(aggressive), Fixed::from_int(0), Fixed::from_int(100));
    country->leader_traits.diplomatic = clamp_fixed(Fixed::from_double(diplomatic), Fixed::from_int(0), Fixed::from_int(100));
    country->leader_traits.corrupt = clamp_fixed(Fixed::from_double(corrupt), Fixed::from_int(0), Fixed::from_int(100));
    country->leader_traits.competent = clamp_fixed(Fixed::from_double(competent), Fixed::from_int(0), Fixed::from_int(100));
    country->leader_tenure_ticks = 0;
}

InternalUnrestStage compute_unrest_stage(const Country& c) {
    const double dissent = c.politics.public_dissent.to_double();
    const double weariness = c.war_weariness.to_double();
    const double fragility =
        (100.0 - c.politics.government_stability.to_double()) * 0.45 +
        (100.0 - c.economic_stability.to_double()) * 0.35 +
        dissent * 0.20;
    const double pressure = dissent * 0.55 + weariness * 0.30 + fragility * 0.15;

    if (pressure < 28.0) {
        return InternalUnrestStage::Calm;
    }
    if (pressure < 45.0) {
        return InternalUnrestStage::Protests;
    }
    if (pressure < 62.0) {
        return InternalUnrestStage::Strikes;
    }
    if (pressure < 80.0) {
        return InternalUnrestStage::Riots;
    }
    return InternalUnrestStage::CivilWar;
}

double perceived_threat_score(const Country& self, const Country& rival) {
    const double self_strength = std::max(1.0, self.military.weighted_total().to_double());
    const double rival_strength = std::max(1.0, rival.military.weighted_total().to_double());
    const double force_ratio = std::clamp(rival_strength / self_strength, 0.2, 2.2);
    const double escalation = std::clamp(rival.escalation_level.to_double() / 5.0, 0.0, 1.0);
    return std::clamp(force_ratio * 0.65 + escalation * 0.35, 0.0, 1.0);
}

double combined_arms_synergy(const MilitaryPower& military) {
    const bool has_infantry = military.units_infantry > Fixed::from_int(45);
    const bool has_armor = military.units_armor > Fixed::from_int(12);
    const bool has_artillery = military.units_artillery > Fixed::from_int(10);
    const bool has_air = military.air_total() > Fixed::from_int(12);
    const bool has_naval = military.naval_total() > Fixed::from_int(10);

    double synergy = 1.0;
    if (has_infantry && has_artillery) {
        synergy *= 1.30;
    }
    if (has_armor && has_air) {
        synergy *= 1.35;
    }
    if (has_naval && has_air) {
        synergy *= 1.25;
    }
    if (has_infantry && has_armor) {
        synergy *= 1.12;
    }
    if (has_artillery && has_air) {
        synergy *= 1.10;
    }

    // Missing components impose coordination penalties on single-domain forces.
    if (has_infantry && !has_artillery) {
        synergy *= 0.92;
    }
    if (has_armor && !has_air) {
        synergy *= 0.90;
    }
    if (has_naval && !has_air) {
        synergy *= 0.94;
    }
    if ((has_infantry || has_armor) && !has_air && !has_artillery) {
        synergy *= 0.90;
    }

    return std::clamp(synergy, 0.70, 1.85);
}

double cobb_douglas_output(double technology_multiplier, double capital, double labor, double alpha) {
    const double safe_a = std::clamp(technology_multiplier, 0.10, 6.0);
    const double safe_k = std::max(1e-3, capital);
    const double safe_l = std::max(1e-3, labor);
    const double safe_alpha = std::clamp(alpha, 0.05, 0.95);
    return safe_a * std::pow(safe_k, safe_alpha) * std::pow(safe_l, 1.0 - safe_alpha);
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
        : width_(width),
            height_(height),
            country_ids_(static_cast<size_t>(width) * static_cast<size_t>(height), 0),
            cell_tags_(static_cast<size_t>(width) * static_cast<size_t>(height), 0),
            sea_zone_ids_(static_cast<size_t>(width) * static_cast<size_t>(height), 0) {}

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

uint8_t GridMap::cell_tags_at(uint32_t x, uint32_t y) const {
    return cell_tags_.at(static_cast<size_t>(y) * width_ + x);
}

void GridMap::set_cell_tags(uint32_t x, uint32_t y, uint8_t tags) {
    cell_tags_.at(static_cast<size_t>(y) * width_ + x) = tags;
}

void GridMap::add_cell_tag(uint32_t x, uint32_t y, uint8_t tag) {
    cell_tags_.at(static_cast<size_t>(y) * width_ + x) = static_cast<uint8_t>(cell_tags_.at(static_cast<size_t>(y) * width_ + x) | tag);
}

uint16_t GridMap::sea_zone_at(uint32_t x, uint32_t y) const {
    return sea_zone_ids_.at(static_cast<size_t>(y) * width_ + x);
}

void GridMap::set_sea_zone(uint32_t x, uint32_t y, uint16_t zone_id) {
    sea_zone_ids_.at(static_cast<size_t>(y) * width_ + x) = zone_id;
}

bool GridMap::is_sea_cell(uint32_t x, uint32_t y) const {
    return (cell_tags_at(x, y) & GridMap::kTagSea) != 0;
}

bool GridMap::has_tag(uint32_t x, uint32_t y, uint8_t tag) const {
    return (cell_tags_at(x, y) & tag) != 0;
}

const std::vector<uint16_t>& GridMap::flattened_country_ids() const {
    return country_ids_;
}

std::vector<uint16_t>& GridMap::flattened_country_ids() {
    return country_ids_;
}

const std::vector<uint8_t>& GridMap::flattened_cell_tags() const {
    return cell_tags_;
}

std::vector<uint8_t>& GridMap::flattened_cell_tags() {
    return cell_tags_;
}

const std::vector<uint16_t>& GridMap::flattened_sea_zone_ids() const {
    return sea_zone_ids_;
}

std::vector<uint16_t>& GridMap::flattened_sea_zone_ids() {
    return sea_zone_ids_;
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
        out.write(reinterpret_cast<const char*>(cell_tags_.data()), static_cast<std::streamsize>(cell_tags_.size() * sizeof(uint8_t)));
        out.write(reinterpret_cast<const char*>(sea_zone_ids_.data()), static_cast<std::streamsize>(sea_zone_ids_.size() * sizeof(uint16_t)));
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
    if (magic != kMapMagic || (version != 1 && version != kMapVersion)) {
        return false;
    }
    const uint64_t expected_cells = static_cast<uint64_t>(width_) * static_cast<uint64_t>(height_);
    if (cells != expected_cells || cells > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    country_ids_.resize(static_cast<size_t>(cells));
    cell_tags_.assign(static_cast<size_t>(cells), 0);
    sea_zone_ids_.assign(static_cast<size_t>(cells), 0);
    if (!country_ids_.empty()) {
        in.read(reinterpret_cast<char*>(country_ids_.data()), static_cast<std::streamsize>(country_ids_.size() * sizeof(uint16_t)));
        if (version >= 2) {
            in.read(reinterpret_cast<char*>(cell_tags_.data()), static_cast<std::streamsize>(cell_tags_.size() * sizeof(uint8_t)));
            in.read(reinterpret_cast<char*>(sea_zone_ids_.data()), static_cast<std::streamsize>(sea_zone_ids_.size() * sizeof(uint16_t)));
        }
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

NegotiationEvent::NegotiationEvent(uint16_t country_a,
                                   uint16_t country_b,
                                   uint16_t proposer_id,
                                   std::string terms_type,
                                   std::string terms_details)
    : country_a_(country_a),
      country_b_(country_b),
      proposer_id_(proposer_id),
      terms_type_(std::move(terms_type)),
      terms_details_(std::move(terms_details)) {}

void NegotiationEvent::execute(World& world) {
    world.resolve_negotiation(country_a_, country_b_, proposer_id_, terms_type_, terms_details_);
}

World::World(uint64_t random_seed, uint64_t tick_seconds)
    : tick_seconds_(tick_seconds == 0 ? 3600 : tick_seconds),
      base_seed_(random_seed == 0 ? 1 : random_seed) {}

void World::set_map(const GridMap& map) {
    map_ = map;
    ensure_map_state_buffers();
    recompute_naval_control();
    recompute_territory_cells();
}

GridMap& World::mutable_map() {
    return map_;
}

const GridMap& World::map() const {
    return map_;
}

void World::reserve_countries(size_t count) {
    countries_.reserve(count);
    country_index_by_id_.reserve(count);
}

void World::add_country(const Country& country) {
    if (countries_structure_locked_) {
        throw std::logic_error("cannot add countries after world simulation has started; reserve and add all countries during setup");
    }
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
    countries_structure_locked_ = true;
    ensure_map_state_buffers();
    while (!events_.empty() && events_.top().tick <= current_tick_) {
        ScheduledEvent item = events_.top();
        events_.pop();
        if (item.event) {
            item.event->execute(*this);
        }
    }

    update_fortification_levels();
    recompute_naval_control();

    update_country_dynamics_parallel();
    update_diplomatic_trust_and_pacts();
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
                                   bool allow_nuclear,
                                   double frontline_momentum,
                                   double attacker_exhaustion,
                                   double defender_exhaustion,
                                   double reinforcement_factor) {
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

    const bool has_land_front = countries_share_land_frontier(attacker_id, defender_id);
    const bool amphibious_assault = !has_land_front;
    if (amphibious_assault && !can_launch_amphibious_assault(attacker_id, defender_id)) {
        attacker->supply_stockpile = clamp_fixed(attacker->supply_stockpile - Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));
        attacker->civilian_morale = clamp_fixed(attacker->civilian_morale - Fixed::from_double(0.2), Fixed::from_int(0), Fixed::from_int(100));
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
    const auto pact_support = [&](const Country& country, uint16_t against_id) {
        std::unordered_map<uint16_t, bool> seen;
        double support = 0.0;

        auto add_supporter = [&](uint16_t supporter_id) {
            if (supporter_id == against_id || supporter_id == country.id || seen[supporter_id]) {
                return;
            }
            seen[supporter_id] = true;
            const Country* supporter = find_country(supporter_id);
            if (supporter == nullptr) {
                return;
            }
            const double trust = bilateral_trust_floor(*supporter, country).to_double();
            if (trust < 20.0) {
                return;
            }
            const double commitment = std::clamp((trust - 20.0) / 80.0, 0.0, 1.0);
            support += supporter->military.weighted_total().to_double() * (0.05 + commitment * 0.19);
        };

        for (uint16_t supporter_id : country.has_defense_pact_with) {
            add_supporter(supporter_id);
        }
        if (country.coalition_id != 0) {
            for (const Country& member : countries_) {
                if (member.id != country.id && member.coalition_id == country.coalition_id) {
                    add_supporter(member.id);
                }
            }
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
    const SupplyRouteAssessment attacker_route = assess_supply_route(attacker_id, defender_id);
    const SupplyRouteAssessment defender_route = assess_supply_route(defender_id, attacker_id);
    const uint32_t effective_route_distance = std::max<uint32_t>(route_distance, attacker_route.route_distance);
    const double attacker_logistics =
        (0.80 + std::min(0.50, attacker->logistics_capacity.to_double() / 180.0)) * attacker_route.efficiency;
    const double defender_logistics =
        (0.80 + std::min(0.50, defender->logistics_capacity.to_double() / 180.0)) * defender_route.efficiency;
    const double attacker_supply = std::clamp(0.68 + attacker->supply_level.to_double() / 140.0 + attacker->supply_capacity.to_double() / 260.0, 0.45, 1.45);
    const double defender_supply = std::clamp(0.68 + defender->supply_level.to_double() / 140.0 + defender->supply_capacity.to_double() / 260.0, 0.45, 1.45);
    const double rough_terrain = std::clamp((defender->terrain.mountains.to_double() + defender->terrain.forests.to_double()) * 0.5, 0.0, 1.0);
    const double urban_terrain = std::clamp(defender->terrain.urban.to_double(), 0.0, 1.0);
    const double open_terrain = std::clamp(1.0 - rough_terrain - urban_terrain * 0.35, 0.2, 1.0);
    const double weather_penalty = 1.0 - std::min(0.35, (attacker->weather_severity.to_double() + defender->weather_severity.to_double()) / 600.0);
    const double season_effect = 0.90 + std::min(0.20, std::abs(attacker->seasonal_effect.to_double() - defender->seasonal_effect.to_double()) / 250.0);
    const double defender_fortification_bonus = frontier_fortification_bonus(defender_id, attacker_id);
    const double attacker_fortification_penalty = frontier_fortification_bonus(attacker_id, defender_id) * 0.45;
    const double momentum_attack_scale = std::clamp(1.0 + frontline_momentum * 0.12, 0.80, 1.32);
    const double momentum_defense_scale = std::clamp(1.0 - frontline_momentum * 0.10, 0.80, 1.28);
    const double attacker_exhaustion_scale = std::clamp(1.0 - attacker_exhaustion * 0.16, 0.72, 1.05);
    const double defender_exhaustion_scale = std::clamp(1.0 - defender_exhaustion * 0.16, 0.72, 1.05);
    const double defender_strategic_bonus = strategic_frontier_bonus(defender_id, attacker_id);
    const double attacker_strategic_bonus = strategic_frontier_bonus(attacker_id, defender_id);
    const double amphibious_penalty = amphibious_assault ? 0.78 : 1.0;
    const double amphibious_surprise = amphibious_assault ? 1.05 : 1.0;
    const double attack_strength =
        (ground_strength(*attacker, *defender, rough_terrain, open_terrain, urban_terrain) +
         air_strength(*attacker, *defender) + naval_strength(*attacker, *defender)) *
        terrain_factor.to_double() * surprise_factor.to_double() * fog_of_war * weather * crit * attacker_intel * attacker_tech *
        attacker_logistics * attacker_supply * combined_arms_synergy(attacker->military) * weather_penalty * momentum_attack_scale *
        attacker_exhaustion_scale * reinforcement_factor * (1.0 - attacker_fortification_penalty) *
        (1.0 + attacker_strategic_bonus * 0.12) * amphibious_penalty * amphibious_surprise;
    const double defense_strength =
        ((ground_strength(*defender, *attacker, rough_terrain, open_terrain, urban_terrain) +
          air_strength(*defender, *attacker) + naval_strength(*defender, *attacker)) *
         (1.16 + defender->terrain.mountains.to_double() * 0.12 + urban_terrain * 0.10) + pact_support(*defender, attacker_id)) *
        (1.20 - std::min(0.80, terrain_factor.to_double() * 0.20)) * (1.0 + 0.10 * unit_dist(rng)) *
        defender_intel * defender_tech * defender_logistics * defender_supply * combined_arms_synergy(defender->military) * season_effect *
        (1.0 + defender_fortification_bonus + defender_strategic_bonus) * momentum_defense_scale * defender_exhaustion_scale;

    const double total_strength = std::max(1e-6, attack_strength + defense_strength);
    const double intensity = 0.08 + 0.04 * unit_dist(rng);

    double attacker_losses_ratio = (defense_strength / total_strength) * intensity;
    double defender_losses_ratio = (attack_strength / total_strength) * intensity;

    const double attacker_attrition =
        std::min(0.30,
                 static_cast<double>(effective_route_distance) * 0.010 +
                     (100.0 - attacker->supply_level.to_double()) / 420.0 +
                     (1.0 - attacker_route.efficiency) * 0.16);
    const double defender_attrition =
        std::min(0.24,
                 static_cast<double>(defender_route.route_distance) * 0.006 +
                     (100.0 - defender->supply_level.to_double()) / 580.0 +
                     (1.0 - defender_route.efficiency) * 0.12);

    bool forced_retreat = false;
    const double attack_to_defense_ratio = attack_strength / std::max(1e-6, defense_strength);
    if (attack_to_defense_ratio > 3.0) {
        const double collapse = sigmoid((attack_to_defense_ratio - 3.0) * 2.4);
        defender_losses_ratio *= 1.0 + collapse * 1.30;
        attacker_losses_ratio *= std::max(0.55, 1.0 - collapse * 0.35);
        forced_retreat = collapse > 0.58;
    }

    const double defense_to_attack_ratio = defense_strength / std::max(1e-6, attack_strength);
    if (defense_to_attack_ratio > 3.0) {
        const double collapse = sigmoid((defense_to_attack_ratio - 3.0) * 2.4);
        attacker_losses_ratio *= 1.0 + collapse * 1.30;
        defender_losses_ratio *= std::max(0.55, 1.0 - collapse * 0.35);
        forced_retreat = collapse > 0.58;
    }

    attacker_losses_ratio = std::clamp(attacker_losses_ratio, 0.0, 0.62);
    defender_losses_ratio = std::clamp(defender_losses_ratio, 0.0, 0.62);

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
    attacker->supply_stockpile = clamp_fixed(attacker->supply_stockpile - Fixed::from_double(2.0 + attacker_attrition * 25.0 + attacker_route.interdiction * 8.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->supply_stockpile = clamp_fixed(defender->supply_stockpile - Fixed::from_double(1.2 + defender_attrition * 20.0 + defender_route.interdiction * 6.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->supply_level = clamp_fixed(attacker->supply_level - Fixed::from_double(2.5 + attacker_attrition * 22.0), Fixed::from_int(0), Fixed::from_int(100));
    defender->supply_level = clamp_fixed(defender->supply_level - Fixed::from_double(1.8 + defender_attrition * 18.0), Fixed::from_int(0), Fixed::from_int(100));
    attacker->recent_combat_losses += Fixed::from_double(attacker_losses_ratio * 100.0 + attacker_attrition * 20.0);
    defender->recent_combat_losses += Fixed::from_double(defender_losses_ratio * 100.0 + defender_attrition * 20.0);
    attacker->war_duration_ticks += 1;
    defender->war_duration_ticks += 1;
    const double attacker_threat = perceived_threat_score(*attacker, *defender);
    const double defender_threat = perceived_threat_score(*defender, *attacker);
    const double attacker_weariness_gain = attacker_losses_ratio * 11.0 + 0.35 + (1.0 - attacker_threat) * 1.15;
    const double defender_weariness_gain = defender_losses_ratio * 11.5 + 0.45 + (1.0 - defender_threat) * 1.10;
    attacker->war_weariness = clamp_fixed(attacker->war_weariness + Fixed::from_double(attacker_weariness_gain), Fixed::from_int(0), Fixed::from_int(100));
    defender->war_weariness = clamp_fixed(defender->war_weariness + Fixed::from_double(defender_weariness_gain), Fixed::from_int(0), Fixed::from_int(100));
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
    if (forced_retreat && attack_to_defense_ratio > 1.0) {
        const uint32_t retreat_cells = static_cast<uint32_t>(std::clamp(2.0 + (attack_to_defense_ratio - 3.0) * 3.5, 2.0, 8.0));
        territory_gain = transfer_border_cells(attacker_id, defender_id, retreat_cells);
    } else if (forced_retreat && defense_to_attack_ratio > 1.0) {
        const uint32_t retreat_cells = static_cast<uint32_t>(std::clamp(2.0 + (defense_to_attack_ratio - 3.0) * 3.5, 2.0, 8.0));
        territory_gain = -transfer_border_cells(defender_id, attacker_id, retreat_cells);
    } else if (defender_losses_ratio > attacker_losses_ratio * 1.15) {
        const double momentum = std::max(0.0, defender_losses_ratio - attacker_losses_ratio);
        const uint32_t max_cells = static_cast<uint32_t>(1 + std::floor(momentum * 40.0));
        territory_gain = transfer_border_cells(attacker_id, defender_id, max_cells);
    } else if (attacker_losses_ratio > defender_losses_ratio * 1.20) {
        const double momentum = std::max(0.0, attacker_losses_ratio - defender_losses_ratio);
        const uint32_t max_cells = static_cast<uint32_t>(1 + std::floor(momentum * 28.0));
        territory_gain = -transfer_border_cells(defender_id, attacker_id, max_cells);
    }

    result.territory_delta = territory_gain;
    if (territory_gain > 0) {
        const Fixed conquest_gain = Fixed::from_double(std::min(8.0, 1.2 + territory_gain * 0.75));
        attacker->resource_oil_reserves = clamp_fixed(attacker->resource_oil_reserves + conquest_gain / Fixed::from_int(4), Fixed::from_int(0), Fixed::from_int(100));
        attacker->resource_minerals_reserves = clamp_fixed(attacker->resource_minerals_reserves + conquest_gain / Fixed::from_int(3), Fixed::from_int(0), Fixed::from_int(100));
        attacker->resource_food_reserves = clamp_fixed(attacker->resource_food_reserves + conquest_gain / Fixed::from_int(4), Fixed::from_int(0), Fixed::from_int(100));
        attacker->resource_rare_earth_reserves = clamp_fixed(attacker->resource_rare_earth_reserves + conquest_gain / Fixed::from_int(5), Fixed::from_int(0), Fixed::from_int(100));
    }
    result.offensive_continues = !result.nuclear_exchange && !forced_retreat && std::abs(territory_gain) < 3;
    return result;
}

CombatResult World::resolve_offensive_tick(uint16_t attacker_id,
                                           uint16_t defender_id,
                                           uint32_t route_distance,
                                           uint32_t remaining_ticks) {
    const uint64_t key = engagement_key(attacker_id, defender_id);
    ActiveEngagement& engagement = active_engagements_[key];
    if (engagement.remaining_ticks == 0) {
        engagement.remaining_ticks = remaining_ticks;
        engagement.route_distance = route_distance;
        engagement.momentum = 0.0;
        engagement.attacker_exhaustion = 0.0;
        engagement.defender_exhaustion = 0.0;
    }

    const Country* attacker = find_country(attacker_id);
    const Country* defender = find_country(defender_id);
    const double attacker_supply = attacker == nullptr ? 0.0 : attacker->supply_level.to_double();
    const double defender_supply = defender == nullptr ? 0.0 : defender->supply_level.to_double();
    const double reinforcement_factor = std::clamp(
        1.0 + (attacker_supply - 50.0) / 250.0 - engagement.attacker_exhaustion * 0.05,
        0.82,
        1.15);

    const double terrain_roll = 0.92 + 0.16 * std::sin(static_cast<double>(current_tick_) * 0.19 + static_cast<double>(attacker_id)) -
                                engagement.momentum * 0.04 + engagement.defender_exhaustion * 0.03;
    const double surprise_roll = 0.90 + 0.22 * std::cos(static_cast<double>(current_tick_) * 0.27 + static_cast<double>(defender_id)) +
                                 engagement.momentum * 0.06;

    CombatResult result = resolve_attack(attacker_id,
                                         defender_id,
                                         Fixed::from_double(std::clamp(terrain_roll, 0.65, 1.20)),
                                         Fixed::from_double(std::clamp(surprise_roll, 0.70, 1.25)),
                                         engagement.route_distance,
                                         true,
                                         engagement.momentum,
                                         engagement.attacker_exhaustion,
                                         engagement.defender_exhaustion,
                                         reinforcement_factor);

    const double attacker_loss = result.attacker_army_losses.to_double() + result.attacker_air_losses.to_double() + result.attacker_navy_losses.to_double();
    const double defender_loss = result.defender_army_losses.to_double() + result.defender_air_losses.to_double() + result.defender_navy_losses.to_double();
    const double exchange = (defender_loss - attacker_loss) / std::max(20.0, attacker_loss + defender_loss);
    engagement.momentum = std::clamp(engagement.momentum * 0.78 + exchange * 0.85, -1.0, 1.0);

    engagement.attacker_exhaustion = std::clamp(
        engagement.attacker_exhaustion + 0.10 + std::max(0.0, attacker_loss - defender_loss) / 160.0 - (attacker_supply / 220.0),
        0.0,
        2.8);
    engagement.defender_exhaustion = std::clamp(
        engagement.defender_exhaustion + 0.10 + std::max(0.0, defender_loss - attacker_loss) / 160.0 - (defender_supply / 230.0),
        0.0,
        2.8);

    if (attacker != nullptr) {
        const Fixed reinf = Fixed::from_double(std::max(0.0, attacker->logistics_capacity.to_double() / 360.0 - engagement.attacker_exhaustion * 0.02));
        Country* mutable_attacker = find_country(attacker_id);
        if (mutable_attacker != nullptr) {
            mutable_attacker->military.units_infantry = clamp_fixed(mutable_attacker->military.units_infantry + reinf,
                                                                     Fixed::from_int(0),
                                                                     Fixed::from_int(1000000));
        }
    }
    if (defender != nullptr) {
        const Fixed reinf = Fixed::from_double(std::max(0.0, defender->logistics_capacity.to_double() / 380.0 - engagement.defender_exhaustion * 0.02));
        Country* mutable_defender = find_country(defender_id);
        if (mutable_defender != nullptr) {
            mutable_defender->military.units_infantry = clamp_fixed(mutable_defender->military.units_infantry + reinf,
                                                                     Fixed::from_int(0),
                                                                     Fixed::from_int(1000000));
        }
    }

    if (engagement.remaining_ticks > 0) {
        --engagement.remaining_ticks;
    }
    const uint32_t next_remaining = std::min<uint32_t>(engagement.remaining_ticks, remaining_ticks > 0 ? remaining_ticks - 1 : 0);

    if (next_remaining > 0 && result.offensive_continues) {
        engagement.route_distance = std::max<uint32_t>(1, engagement.route_distance + 1);
        schedule_event(std::make_unique<OffensiveEvent>(attacker_id, defender_id, next_remaining, engagement.route_distance),
                       current_tick_ + 1);
    } else {
        active_engagements_.erase(key);
    }
    return result;
}

uint32_t World::negotiation_key(uint16_t country_a, uint16_t country_b) const {
    const uint16_t lo = std::min(country_a, country_b);
    const uint16_t hi = std::max(country_a, country_b);
    return (static_cast<uint32_t>(lo) << 16U) | static_cast<uint32_t>(hi);
}

void World::resolve_negotiation(uint16_t country_a, uint16_t country_b) {
    resolve_negotiation(country_a, country_b, country_a, "ceasefire", "");
}

void World::resolve_negotiation(uint16_t country_a,
                               uint16_t country_b,
                               uint16_t proposer_id,
                               const std::string& terms_type,
                               const std::string& terms_details) {
    Country* a = find_country(country_a);
    Country* b = find_country(country_b);
    if (a == nullptr || b == nullptr || country_a == country_b) {
        return;
    }

    const uint32_t key = negotiation_key(country_a, country_b);
    const bool explicit_proposal = proposer_id != 0 || !terms_type.empty() || !terms_details.empty();

    auto parse_territory_offer = [](const std::string& details) {
        const std::string needle = "territory";
        std::string lowered = details;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const size_t pos = lowered.find(needle);
        if (pos == std::string::npos) {
            return 0;
        }
        size_t i = pos + needle.size();
        while (i < lowered.size() && !std::isdigit(static_cast<unsigned char>(lowered[i]))) {
            ++i;
        }
        if (i >= lowered.size()) {
            return 0;
        }
        int value = 0;
        while (i < lowered.size() && std::isdigit(static_cast<unsigned char>(lowered[i]))) {
            value = value * 10 + (lowered[i] - '0');
            ++i;
        }
        return std::clamp(value, 0, 8);
    };

    auto session_it = negotiation_sessions_.find(key);
    if (session_it == negotiation_sessions_.end() || explicit_proposal) {
        NegotiationSession session;
        session.country_a = country_a;
        session.country_b = country_b;
        session.proposer_id = proposer_id == 0 ? country_a : proposer_id;
        session.round = 1;
        session.stage = 0;
        session.terms_type = terms_type.empty() ? "ceasefire" : terms_type;
        session.terms_details = terms_details;
        session.last_updated_tick = current_tick_;
        negotiation_sessions_[key] = session;

        a->trust_scores[country_b] = clamp_fixed(a->trust_scores[country_b] + Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));
        b->trust_scores[country_a] = clamp_fixed(b->trust_scores[country_a] + Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));

        schedule_event(std::make_unique<NegotiationEvent>(country_a, country_b, 0, "", ""), current_tick_ + 1);
        return;
    }

    NegotiationSession& session = session_it->second;
    if (session.round >= 6) {
        a->trust_scores[country_b] = clamp_fixed(a->trust_scores[country_b] - Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
        b->trust_scores[country_a] = clamp_fixed(b->trust_scores[country_a] - Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
        negotiation_sessions_.erase(session_it);
        return;
    }

    Country* proposer = find_country(session.proposer_id);
    if (proposer == nullptr) {
        proposer = a;
        session.proposer_id = proposer->id;
    }
    Country* responder = proposer->id == a->id ? b : a;

    const double trust = bilateral_trust_floor(*a, *b).to_double();

    if (session.stage == 0) {
        if (trust < 15.0) {
            a->trust_scores[b->id] = clamp_fixed(a->trust_scores[b->id] - Fixed::from_double(1.8), Fixed::from_int(0), Fixed::from_int(100));
            b->trust_scores[a->id] = clamp_fixed(b->trust_scores[a->id] - Fixed::from_double(1.8), Fixed::from_int(0), Fixed::from_int(100));
            negotiation_sessions_.erase(session_it);
            return;
        }

        if (session.terms_type == "defense_pact" && trust <= 70.0) {
            session.terms_type = "non_aggression";
            if (session.terms_details.empty()) {
                session.terms_details = "counter: non-aggression until trust improves";
            }
        } else if (session.terms_type == "alliance" && trust < 55.0) {
            session.terms_type = "non_aggression";
            if (session.terms_details.empty()) {
                session.terms_details = "counter: confidence-building pact first";
            }
        } else if (session.terms_type == "ceasefire" && session.terms_details.empty()) {
            const double proposer_strength = proposer->military.weighted_total().to_double();
            const double responder_strength = responder->military.weighted_total().to_double();
            if (proposer_strength > responder_strength * 1.12) {
                session.terms_details = "territory:1";
            }
        }

        session.stage = 1;
        ++session.round;
        session.last_updated_tick = current_tick_;
        schedule_event(std::make_unique<NegotiationEvent>(country_a, country_b, 0, "", ""), current_tick_ + 1);
        return;
    }

    const double relationship_quality = 0.5 * (a->reputation.to_double() + b->reputation.to_double());
    const double accept_score = trust * 0.65 + relationship_quality * 0.35;
    bool accepted = false;

    if (session.terms_type == "defense_pact") {
        accepted = trust > 70.0 && accept_score > 62.0;
    } else if (session.terms_type == "alliance") {
        accepted = trust > 58.0 && accept_score > 55.0;
    } else if (session.terms_type == "non_aggression") {
        accepted = trust >= 20.0 && accept_score > 35.0;
    } else if (session.terms_type == "betray") {
        accepted = false;
    } else {
        accepted = accept_score > 32.0;
    }

    if (!accepted) {
        a->trust_scores[b->id] = clamp_fixed(a->trust_scores[b->id] - Fixed::from_double(1.0), Fixed::from_int(0), Fixed::from_int(100));
        b->trust_scores[a->id] = clamp_fixed(b->trust_scores[a->id] - Fixed::from_double(1.0), Fixed::from_int(0), Fixed::from_int(100));
        negotiation_sessions_.erase(session_it);
        return;
    }

    a->civilian_morale = clamp_fixed(a->civilian_morale + Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
    b->civilian_morale = clamp_fixed(b->civilian_morale + Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
    a->economic_stability = clamp_fixed(a->economic_stability + Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));
    b->economic_stability = clamp_fixed(b->economic_stability + Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));
    a->trust_scores[b->id] = clamp_fixed(a->trust_scores[b->id] + Fixed::from_double(2.6), Fixed::from_int(0), Fixed::from_int(100));
    b->trust_scores[a->id] = clamp_fixed(b->trust_scores[a->id] + Fixed::from_double(2.6), Fixed::from_int(0), Fixed::from_int(100));

    if (session.terms_type == "alliance") {
        merge_or_create_coalition(a->id, b->id);
    }
    if (session.terms_type == "betray") {
        register_betrayal(session.proposer_id, responder->id, Fixed::from_double(16.0));
    }

    if (session.terms_type == "defense_pact" || session.terms_type == "alliance") {
        if (bilateral_trust_floor(*a, *b).to_double() > 70.0) {
            pallas::util::add_unique_id(&a->has_defense_pact_with, b->id);
            pallas::util::add_unique_id(&b->has_defense_pact_with, a->id);
            a->defense_pact_expiry_ticks[b->id] = current_tick_ + 36;
            b->defense_pact_expiry_ticks[a->id] = current_tick_ + 36;
        }
    }

    if (session.terms_type == "non_aggression" || session.terms_type == "ceasefire" || session.terms_type == "alliance") {
        if (bilateral_trust_floor(*a, *b).to_double() >= 20.0) {
            pallas::util::add_unique_id(&a->has_non_aggression_with, b->id);
            pallas::util::add_unique_id(&b->has_non_aggression_with, a->id);
            a->non_aggression_expiry_ticks[b->id] = current_tick_ + 24;
            b->non_aggression_expiry_ticks[a->id] = current_tick_ + 24;
        }
    }

    if (!is_embargoed_between(*a, *b)) {
        pallas::util::add_unique_id(&a->trade_partners, b->id);
        pallas::util::add_unique_id(&b->trade_partners, a->id);
    }

    const int territory_offer = parse_territory_offer(session.terms_details);
    if (session.terms_type == "ceasefire" && territory_offer > 0) {
        transfer_border_cells(proposer->id, responder->id, static_cast<uint32_t>(territory_offer));
    }

    if (a->diplomatic_stance == DiplomaticStance::Aggressive) {
        a->diplomatic_stance = DiplomaticStance::Neutral;
    }
    if (b->diplomatic_stance == DiplomaticStance::Aggressive) {
        b->diplomatic_stance = DiplomaticStance::Neutral;
    }

    enforce_pact_gates(a, b);
    negotiation_sessions_.erase(session_it);
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
    ensure_map_state_buffers();
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

void World::ensure_map_state_buffers() {
    const size_t cell_count = map_.flattened_country_ids().size();
    if (fortification_level_per_cell_.size() != cell_count) {
        fortification_level_per_cell_.assign(cell_count, 0);
    }
    if (previous_owner_per_cell_.size() != cell_count) {
        previous_owner_per_cell_ = map_.flattened_country_ids();
    }
    if (map_.flattened_cell_tags().size() != cell_count) {
        map_.flattened_cell_tags().assign(cell_count, 0);
    }
    if (map_.flattened_sea_zone_ids().size() != cell_count) {
        map_.flattened_sea_zone_ids().assign(cell_count, 0);
    }
}

void World::update_fortification_levels() {
    ensure_map_state_buffers();
    const std::vector<uint16_t>& owners = map_.flattened_country_ids();
    if (owners.empty()) {
        return;
    }

    for (size_t i = 0; i < owners.size(); ++i) {
        const uint16_t owner = owners[i];
        uint16_t& fort = fortification_level_per_cell_[i];
        if (owner == 0) {
            fort = 0;
            previous_owner_per_cell_[i] = owner;
            continue;
        }

        if (owner == previous_owner_per_cell_[i]) {
            fort = static_cast<uint16_t>(std::min<int>(400, static_cast<int>(fort) + 15));
        } else {
            fort = static_cast<uint16_t>(std::max<int>(0, static_cast<int>(fort) - 160));
        }
        previous_owner_per_cell_[i] = owner;
    }
}

double World::frontier_fortification_bonus(uint16_t owner_id, uint16_t against_id) const {
    if (map_.width() == 0 || map_.height() == 0 || fortification_level_per_cell_.empty()) {
        return 0.0;
    }

    int64_t sum = 0;
    uint32_t count = 0;
    const uint32_t width = map_.width();
    const uint32_t height = map_.height();

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (map_.at(x, y) != owner_id) {
                continue;
            }

            bool frontier = false;
            if (x > 0 && map_.at(x - 1, y) == against_id) {
                frontier = true;
            }
            if (!frontier && x + 1 < width && map_.at(x + 1, y) == against_id) {
                frontier = true;
            }
            if (!frontier && y > 0 && map_.at(x, y - 1) == against_id) {
                frontier = true;
            }
            if (!frontier && y + 1 < height && map_.at(x, y + 1) == against_id) {
                frontier = true;
            }
            if (!frontier) {
                continue;
            }

            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + x;
            if (idx >= fortification_level_per_cell_.size()) {
                continue;
            }
            sum += fortification_level_per_cell_[idx];
            ++count;
        }
    }

    if (count == 0) {
        return 0.0;
    }
    const double avg_milli = static_cast<double>(sum) / static_cast<double>(count);
    return std::clamp(avg_milli / 1000.0, 0.0, 0.40);
}

SupplyRouteAssessment World::assess_supply_route(uint16_t country_id, uint16_t enemy_id) const {
    SupplyRouteAssessment out;
    if (map_.width() == 0 || map_.height() == 0) {
        return out;
    }

    const Country* country = find_country(country_id);
    if (country == nullptr) {
        out.efficiency = 0.30;
        out.encircled = true;
        out.route_distance = 12;
        return out;
    }

    const uint32_t width = map_.width();
    const uint32_t height = map_.height();
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (total == 0) {
        return out;
    }

    auto idx_of = [width](uint32_t x, uint32_t y) {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    std::vector<uint8_t> frontier(total, 0);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (map_.at(x, y) != country_id) {
                continue;
            }
            bool is_frontier = false;
            if (x > 0 && map_.at(x - 1, y) == enemy_id) {
                is_frontier = true;
            }
            if (!is_frontier && x + 1 < width && map_.at(x + 1, y) == enemy_id) {
                is_frontier = true;
            }
            if (!is_frontier && y > 0 && map_.at(x, y - 1) == enemy_id) {
                is_frontier = true;
            }
            if (!is_frontier && y + 1 < height && map_.at(x, y + 1) == enemy_id) {
                is_frontier = true;
            }
            if (is_frontier) {
                frontier[idx_of(x, y)] = 1;
            }
        }
    }

    bool has_frontier = false;
    for (uint8_t f : frontier) {
        if (f != 0) {
            has_frontier = true;
            break;
        }
    }
    if (!has_frontier) {
        out.route_distance = 1;
        out.efficiency = 1.0;
        out.interdiction = 0.0;
        return out;
    }

    std::vector<size_t> start_nodes;
    if (country->capital.x >= 0 && country->capital.y >= 0 &&
        static_cast<uint32_t>(country->capital.x) < width && static_cast<uint32_t>(country->capital.y) < height) {
        const size_t capital_idx = idx_of(static_cast<uint32_t>(country->capital.x), static_cast<uint32_t>(country->capital.y));
        if (map_.flattened_country_ids()[capital_idx] == country_id) {
            start_nodes.push_back(capital_idx);
        }
    }

    // Optional forward depot: most fortified owned cell provides alternate source.
    size_t depot_idx = std::numeric_limits<size_t>::max();
    uint16_t best_fort = 0;
    for (size_t i = 0; i < map_.flattened_country_ids().size(); ++i) {
        if (map_.flattened_country_ids()[i] != country_id || i >= fortification_level_per_cell_.size()) {
            continue;
        }
        if (fortification_level_per_cell_[i] > best_fort) {
            best_fort = fortification_level_per_cell_[i];
            depot_idx = i;
        }
    }
    if (depot_idx != std::numeric_limits<size_t>::max() && best_fort >= 120) {
        start_nodes.push_back(depot_idx);
    }

    if (start_nodes.empty()) {
        for (size_t i = 0; i < map_.flattened_country_ids().size(); ++i) {
            if (map_.flattened_country_ids()[i] == country_id) {
                start_nodes.push_back(i);
                break;
            }
        }
    }

    if (start_nodes.empty()) {
        out.efficiency = 0.30;
        out.encircled = true;
        out.route_distance = 12;
        return out;
    }

    std::vector<int32_t> dist(total, -1);
    std::vector<size_t> parent(total, std::numeric_limits<size_t>::max());
    std::deque<size_t> q;

    for (size_t start : start_nodes) {
        dist[start] = 0;
        q.push_back(start);
    }

    size_t goal = std::numeric_limits<size_t>::max();
    while (!q.empty()) {
        const size_t cur = q.front();
        q.pop_front();
        if (frontier[cur] != 0) {
            goal = cur;
            break;
        }

        const uint32_t x = static_cast<uint32_t>(cur % width);
        const uint32_t y = static_cast<uint32_t>(cur / width);
        const uint32_t nx[4] = {x > 0 ? x - 1 : x, x + 1 < width ? x + 1 : x, x, x};
        const uint32_t ny[4] = {y, y, y > 0 ? y - 1 : y, y + 1 < height ? y + 1 : y};

        for (int k = 0; k < 4; ++k) {
            if (nx[k] == x && ny[k] == y) {
                continue;
            }
            const size_t ni = idx_of(nx[k], ny[k]);
            if (dist[ni] >= 0) {
                continue;
            }
            if (map_.flattened_country_ids()[ni] != country_id) {
                continue;
            }
            dist[ni] = dist[cur] + 1;
            parent[ni] = cur;
            q.push_back(ni);
        }
    }

    if (goal == std::numeric_limits<size_t>::max()) {
        out.efficiency = 0.30;
        out.interdiction = 1.0;
        out.encircled = true;
        out.route_distance = 14;
        return out;
    }

    int threatened_cells = 0;
    int path_len = 0;
    for (size_t cur = goal; cur != std::numeric_limits<size_t>::max(); cur = parent[cur]) {
        const uint32_t x = static_cast<uint32_t>(cur % width);
        const uint32_t y = static_cast<uint32_t>(cur / width);
        bool threatened = false;
        if (x > 0 && map_.at(x - 1, y) == enemy_id) {
            threatened = true;
        }
        if (!threatened && x + 1 < width && map_.at(x + 1, y) == enemy_id) {
            threatened = true;
        }
        if (!threatened && y > 0 && map_.at(x, y - 1) == enemy_id) {
            threatened = true;
        }
        if (!threatened && y + 1 < height && map_.at(x, y + 1) == enemy_id) {
            threatened = true;
        }
        threatened_cells += threatened ? 1 : 0;
        ++path_len;
    }

    out.route_distance = static_cast<uint32_t>(std::max(1, path_len));
    out.interdiction = path_len > 0 ? static_cast<double>(threatened_cells) / static_cast<double>(path_len) : 0.0;
    const double distance_penalty = std::max(0.75, 1.0 - static_cast<double>(out.route_distance) * 0.010);
    const double interdiction_penalty = std::max(0.30, 1.0 - out.interdiction * 0.70);
    out.efficiency = std::clamp(distance_penalty * interdiction_penalty, 0.30, 1.10);
    out.encircled = out.efficiency <= 0.31;
    return out;
}

uint64_t World::engagement_key(uint16_t attacker_id, uint16_t defender_id) const {
    return (static_cast<uint64_t>(attacker_id) << 32U) | static_cast<uint64_t>(defender_id);
}

int32_t World::transfer_border_cells(uint16_t attacker_id, uint16_t defender_id, uint32_t max_cells) {
    if (map_.width() == 0 || map_.height() == 0 || max_cells == 0) {
        return 0;
    }

    uint32_t changed = 0;
    for (uint32_t y = 0; y < map_.height() && changed < max_cells; ++y) {
        for (uint32_t x = 0; x < map_.width() && changed < max_cells; ++x) {
            if (map_.at(x, y) != defender_id || map_.is_sea_cell(x, y)) {
                continue;
            }
            bool has_attacker_neighbor = false;
            if (x > 0 && map_.at(x - 1, y) == attacker_id && !map_.is_sea_cell(x - 1, y)) {
                has_attacker_neighbor = true;
            }
            if (!has_attacker_neighbor && x + 1 < map_.width() && map_.at(x + 1, y) == attacker_id && !map_.is_sea_cell(x + 1, y)) {
                has_attacker_neighbor = true;
            }
            if (!has_attacker_neighbor && y > 0 && map_.at(x, y - 1) == attacker_id && !map_.is_sea_cell(x, y - 1)) {
                has_attacker_neighbor = true;
            }
            if (!has_attacker_neighbor && y + 1 < map_.height() && map_.at(x, y + 1) == attacker_id && !map_.is_sea_cell(x, y + 1)) {
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

bool World::countries_share_land_frontier(uint16_t country_a, uint16_t country_b) const {
    if (country_a == 0 || country_b == 0 || country_a == country_b) {
        return false;
    }
    if (map_.width() == 0 || map_.height() == 0) {
        return false;
    }
    const uint32_t width = map_.width();
    const uint32_t height = map_.height();
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (map_.at(x, y) != country_a || map_.is_sea_cell(x, y)) {
                continue;
            }
            if (x > 0 && map_.at(x - 1, y) == country_b && !map_.is_sea_cell(x - 1, y)) {
                return true;
            }
            if (x + 1 < width && map_.at(x + 1, y) == country_b && !map_.is_sea_cell(x + 1, y)) {
                return true;
            }
            if (y > 0 && map_.at(x, y - 1) == country_b && !map_.is_sea_cell(x, y - 1)) {
                return true;
            }
            if (y + 1 < height && map_.at(x, y + 1) == country_b && !map_.is_sea_cell(x, y + 1)) {
                return true;
            }
        }
    }
    return false;
}

std::vector<uint16_t> World::coastal_sea_zones(uint16_t country_id) const {
    std::unordered_set<uint16_t> zones;
    if (country_id == 0 || map_.width() == 0 || map_.height() == 0) {
        return {};
    }

    const uint32_t width = map_.width();
    const uint32_t height = map_.height();
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (map_.at(x, y) != country_id || map_.is_sea_cell(x, y)) {
                continue;
            }

            auto add_zone = [&](uint32_t nx, uint32_t ny) {
                if (!map_.is_sea_cell(nx, ny)) {
                    return;
                }
                const uint16_t zone = map_.sea_zone_at(nx, ny);
                if (zone != 0) {
                    zones.insert(zone);
                }
            };
            if (x > 0) add_zone(x - 1, y);
            if (x + 1 < width) add_zone(x + 1, y);
            if (y > 0) add_zone(x, y - 1);
            if (y + 1 < height) add_zone(x, y + 1);
        }
    }

    std::vector<uint16_t> out(zones.begin(), zones.end());
    std::sort(out.begin(), out.end());
    return out;
}

void World::recompute_naval_control() {
    dominant_controller_by_sea_zone_.clear();
    naval_dominance_ratio_by_sea_zone_.clear();
    blockade_pressure_by_country_.clear();
    strategic_economic_bonus_by_country_.clear();

    if (map_.width() == 0 || map_.height() == 0) {
        return;
    }

    std::unordered_map<uint16_t, std::unordered_set<uint16_t>> zone_neighbors;
    std::unordered_map<uint16_t, bool> zone_has_chokepoint;
    const uint32_t width = map_.width();
    const uint32_t height = map_.height();

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (!map_.is_sea_cell(x, y)) {
                continue;
            }
            const uint16_t zone = map_.sea_zone_at(x, y);
            if (zone == 0) {
                continue;
            }

            const uint8_t tags = map_.cell_tags_at(x, y);
            if ((tags & GridMap::kTagChokepointStrait) != 0 || (tags & GridMap::kTagChokepointCanal) != 0) {
                zone_has_chokepoint[zone] = true;
            }

            auto add_neighbor = [&](uint32_t nx, uint32_t ny) {
                if (map_.is_sea_cell(nx, ny)) {
                    return;
                }
                const uint16_t owner = map_.at(nx, ny);
                if (owner != 0) {
                    zone_neighbors[zone].insert(owner);
                }
            };

            if (x > 0) add_neighbor(x - 1, y);
            if (x + 1 < width) add_neighbor(x + 1, y);
            if (y > 0) add_neighbor(x, y - 1);
            if (y + 1 < height) add_neighbor(x, y + 1);
        }
    }

    for (const auto& zone_entry : zone_neighbors) {
        const uint16_t zone = zone_entry.first;
        double best = 0.0;
        double second = 0.0;
        uint16_t best_country = 0;
        const bool chokepoint = zone_has_chokepoint[zone];

        for (uint16_t country_id : zone_entry.second) {
            const Country* country = find_country(country_id);
            if (country == nullptr) {
                continue;
            }

            const double naval_surface = country->military.units_naval_surface.to_double();
            const double naval_sub = country->military.units_naval_submarine.to_double();
            const double naval_air_cover = country->military.units_air_fighter.to_double() * 0.26;
            const double intel_edge = country->intelligence_level.to_double() * 0.08;

            int ports = 0;
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    if (map_.at(x, y) != country_id || map_.is_sea_cell(x, y)) {
                        continue;
                    }
                    if ((map_.cell_tags_at(x, y) & GridMap::kTagPort) == 0) {
                        continue;
                    }
                    bool touches_zone = false;
                    if (x > 0 && map_.is_sea_cell(x - 1, y) && map_.sea_zone_at(x - 1, y) == zone) touches_zone = true;
                    if (!touches_zone && x + 1 < width && map_.is_sea_cell(x + 1, y) && map_.sea_zone_at(x + 1, y) == zone) touches_zone = true;
                    if (!touches_zone && y > 0 && map_.is_sea_cell(x, y - 1) && map_.sea_zone_at(x, y - 1) == zone) touches_zone = true;
                    if (!touches_zone && y + 1 < height && map_.is_sea_cell(x, y + 1) && map_.sea_zone_at(x, y + 1) == zone) touches_zone = true;
                    if (touches_zone) {
                        ++ports;
                    }
                }
            }

            const double port_bonus = std::min(0.45, static_cast<double>(ports) * 0.12);
            const double projection =
                naval_surface * (1.08 + port_bonus) +
                naval_sub * (chokepoint ? 1.52 : 1.24) +
                naval_air_cover +
                intel_edge;

            if (projection > best) {
                second = best;
                best = projection;
                best_country = country_id;
            } else if (projection > second) {
                second = projection;
            }
        }

        if (best_country == 0 || best <= 1e-6) {
            continue;
        }

        const double ratio = std::clamp((best - second) / std::max(1.0, best), 0.0, 1.0);
        if (best > second * 1.15) {
            dominant_controller_by_sea_zone_[zone] = best_country;
            naval_dominance_ratio_by_sea_zone_[zone] = ratio;
        }
    }

    for (const Country& c : countries_) {
        blockade_pressure_by_country_[c.id] = blockade_pressure_against(c.id);
        strategic_economic_bonus_by_country_[c.id] = strategic_economic_bonus(c.id);
    }
}

uint16_t World::dominant_controller_for_zone(uint16_t zone_id, double* dominance_ratio) const {
    if (dominance_ratio != nullptr) {
        *dominance_ratio = 0.0;
    }
    auto it = dominant_controller_by_sea_zone_.find(zone_id);
    if (it == dominant_controller_by_sea_zone_.end()) {
        return 0;
    }
    if (dominance_ratio != nullptr) {
        auto ratio_it = naval_dominance_ratio_by_sea_zone_.find(zone_id);
        if (ratio_it != naval_dominance_ratio_by_sea_zone_.end()) {
            *dominance_ratio = ratio_it->second;
        }
    }
    return it->second;
}

bool World::can_launch_amphibious_assault(uint16_t attacker_id, uint16_t defender_id) const {
    const std::vector<uint16_t> attacker_zones = coastal_sea_zones(attacker_id);
    const std::vector<uint16_t> defender_zones = coastal_sea_zones(defender_id);
    if (attacker_zones.empty() || defender_zones.empty()) {
        return false;
    }

    std::unordered_set<uint16_t> defender_zone_set(defender_zones.begin(), defender_zones.end());
    for (uint16_t zone : attacker_zones) {
        if (defender_zone_set.find(zone) == defender_zone_set.end()) {
            continue;
        }
        double dominance = 0.0;
        const uint16_t controller = dominant_controller_for_zone(zone, &dominance);
        if (controller == attacker_id && dominance >= 0.12) {
            return true;
        }
    }
    return false;
}

double World::strategic_frontier_bonus(uint16_t owner_id, uint16_t against_id) const {
    if (map_.width() == 0 || map_.height() == 0) {
        return 0.0;
    }

    const uint32_t width = map_.width();
    const uint32_t height = map_.height();
    double score = 0.0;
    uint32_t count = 0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (map_.at(x, y) != owner_id || map_.is_sea_cell(x, y)) {
                continue;
            }
            bool frontier = false;
            if (x > 0 && map_.at(x - 1, y) == against_id && !map_.is_sea_cell(x - 1, y)) frontier = true;
            if (!frontier && x + 1 < width && map_.at(x + 1, y) == against_id && !map_.is_sea_cell(x + 1, y)) frontier = true;
            if (!frontier && y > 0 && map_.at(x, y - 1) == against_id && !map_.is_sea_cell(x, y - 1)) frontier = true;
            if (!frontier && y + 1 < height && map_.at(x, y + 1) == against_id && !map_.is_sea_cell(x, y + 1)) frontier = true;
            if (!frontier) {
                continue;
            }

            const uint8_t tags = map_.cell_tags_at(x, y);
            double cell = 0.0;
            if ((tags & GridMap::kTagStrategic) != 0) cell += 0.55;
            if ((tags & GridMap::kTagMountainPass) != 0) cell += 0.42;
            if ((tags & GridMap::kTagRiverCrossing) != 0) cell += 0.32;
            if ((tags & GridMap::kTagPort) != 0) cell += 0.20;
            if ((tags & GridMap::kTagChokepointStrait) != 0 || (tags & GridMap::kTagChokepointCanal) != 0) {
                cell += 0.30;
            }
            score += cell;
            ++count;
        }
    }
    if (count == 0) {
        return 0.0;
    }
    return std::clamp(score / static_cast<double>(count), 0.0, 0.45);
}

double World::strategic_economic_bonus(uint16_t owner_id) const {
    if (map_.width() == 0 || map_.height() == 0) {
        return 0.0;
    }

    const Country* country = find_country(owner_id);
    const double territory = country == nullptr ? 1.0 : std::max(1.0, static_cast<double>(country->territory_cells));
    const uint32_t width = map_.width();
    const uint32_t height = map_.height();
    double weighted = 0.0;

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (map_.at(x, y) != owner_id) {
                continue;
            }
            const uint8_t tags = map_.cell_tags_at(x, y);
            if ((tags & GridMap::kTagStrategic) != 0) weighted += 1.0;
            if ((tags & GridMap::kTagPort) != 0) weighted += 1.4;
            if ((tags & GridMap::kTagRiverCrossing) != 0) weighted += 0.6;
            if ((tags & GridMap::kTagMountainPass) != 0) weighted += 0.7;
        }
    }
    return std::clamp((weighted / territory) * 0.65, 0.0, 1.0);
}

bool World::has_hostile_relationship(uint16_t controller_id, uint16_t target_id) const {
    if (controller_id == 0 || target_id == 0 || controller_id == target_id) {
        return false;
    }
    const Country* controller = find_country(controller_id);
    const Country* target = find_country(target_id);
    if (controller == nullptr || target == nullptr) {
        return true;
    }
    if (pallas::util::contains_id(controller->allied_country_ids, target_id) ||
        pallas::util::contains_id(controller->has_non_aggression_with, target_id) ||
        pallas::util::contains_id(controller->has_trade_treaty_with, target_id) ||
        pallas::util::contains_id(target->allied_country_ids, controller_id) ||
        pallas::util::contains_id(target->has_non_aggression_with, controller_id) ||
        pallas::util::contains_id(target->has_trade_treaty_with, controller_id)) {
        return false;
    }
    return bilateral_trust_floor(*controller, *target).to_double() < 35.0;
}

double World::blockade_pressure_against(uint16_t country_id) const {
    const std::vector<uint16_t> zones = coastal_sea_zones(country_id);
    if (zones.empty()) {
        return 0.0;
    }

    double cumulative = 0.0;
    uint32_t blocked = 0;
    for (uint16_t zone : zones) {
        double dominance = 0.0;
        const uint16_t controller = dominant_controller_for_zone(zone, &dominance);
        if (controller == 0 || controller == country_id) {
            continue;
        }
        if (!has_hostile_relationship(controller, country_id)) {
            continue;
        }
        cumulative += 0.40 + std::min(0.60, dominance * 0.75);
        ++blocked;
    }

    if (blocked == 0) {
        return 0.0;
    }
    return std::clamp(cumulative / static_cast<double>(blocked), 0.0, 1.0);
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

Fixed World::bilateral_trust(const Country& from, uint16_t other_id) const {
    const auto it = from.trust_scores.find(other_id);
    if (it != from.trust_scores.end()) {
        return clamp_fixed(it->second, Fixed::from_int(0), Fixed::from_int(100));
    }
    return Fixed::from_int(50);
}

Fixed World::bilateral_trust_floor(const Country& a, const Country& b) const {
    const Fixed ab = bilateral_trust(a, b.id);
    const Fixed ba = bilateral_trust(b, a.id);
    return ab < ba ? ab : ba;
}

void World::enforce_pact_gates(Country* a, Country* b) {
    if (a == nullptr || b == nullptr) {
        return;
    }

    const double trust = bilateral_trust_floor(*a, *b).to_double();
    if (trust <= 70.0) {
        pallas::util::erase_id(&a->has_defense_pact_with, b->id);
        pallas::util::erase_id(&b->has_defense_pact_with, a->id);
        a->defense_pact_expiry_ticks.erase(b->id);
        b->defense_pact_expiry_ticks.erase(a->id);
    }
    if (trust < 20.0) {
        pallas::util::erase_id(&a->has_non_aggression_with, b->id);
        pallas::util::erase_id(&b->has_non_aggression_with, a->id);
        a->non_aggression_expiry_ticks.erase(b->id);
        b->non_aggression_expiry_ticks.erase(a->id);
    }
}

void World::align_coalition_allies() {
    std::unordered_map<uint16_t, std::vector<uint16_t>> members_by_coalition;
    for (const Country& country : countries_) {
        if (country.coalition_id != 0) {
            members_by_coalition[country.coalition_id].push_back(country.id);
        }
    }

    for (Country& country : countries_) {
        if (country.coalition_id == 0) {
            continue;
        }
        const auto it = members_by_coalition.find(country.coalition_id);
        if (it == members_by_coalition.end()) {
            continue;
        }
        for (uint16_t member_id : it->second) {
            if (member_id != country.id) {
                pallas::util::add_unique_id(&country.allied_country_ids, member_id);
            }
        }
    }
}

void World::merge_or_create_coalition(uint16_t country_a, uint16_t country_b) {
    Country* a = find_country(country_a);
    Country* b = find_country(country_b);
    if (a == nullptr || b == nullptr) {
        return;
    }

    if (a->coalition_id == 0 && b->coalition_id == 0) {
        const uint16_t coalition = std::max<uint16_t>(1, next_coalition_id_++);
        a->coalition_id = coalition;
        b->coalition_id = coalition;
        align_coalition_allies();
        return;
    }

    if (a->coalition_id == 0) {
        a->coalition_id = b->coalition_id;
        align_coalition_allies();
        return;
    }
    if (b->coalition_id == 0) {
        b->coalition_id = a->coalition_id;
        align_coalition_allies();
        return;
    }

    if (a->coalition_id != b->coalition_id) {
        const uint16_t keep = std::min(a->coalition_id, b->coalition_id);
        const uint16_t merge = std::max(a->coalition_id, b->coalition_id);
        for (Country& country : countries_) {
            if (country.coalition_id == merge) {
                country.coalition_id = keep;
            }
        }
        align_coalition_allies();
    }
}

void World::register_betrayal(uint16_t betrayer_id, uint16_t victim_id, Fixed severity) {
    Country* betrayer = find_country(betrayer_id);
    if (betrayer != nullptr) {
        betrayer->betrayal_tick_log.push_back(current_tick_);
    }
    betrayal_history_by_betrayer_[betrayer_id].push_back(current_tick_);

    const double base_drop = std::clamp(severity.to_double(), 2.0, 30.0);
    for (Country& observer : countries_) {
        double multiplier = 0.42;
        if (observer.id == victim_id) {
            multiplier = 1.0;
        } else if (observer.id == betrayer_id) {
            multiplier = 0.24;
        }
        observer.trust_scores[betrayer_id] = clamp_fixed(
            observer.trust_scores[betrayer_id] - Fixed::from_double(base_drop * multiplier),
            Fixed::from_int(0),
            Fixed::from_int(100));
    }
}

double World::betrayal_decay_score(uint16_t betrayer_id) const {
    const auto it = betrayal_history_by_betrayer_.find(betrayer_id);
    if (it == betrayal_history_by_betrayer_.end()) {
        return 0.0;
    }

    constexpr double kHalfLifeTicks = 30.0;
    double decayed = 0.0;
    for (uint64_t tick : it->second) {
        const uint64_t age = current_tick_ > tick ? (current_tick_ - tick) : 0;
        decayed += std::exp2(-static_cast<double>(age) / kHalfLifeTicks);
    }

    // Keep a persistent distrust component so serial betrayers stay suspect.
    const double persistent = std::min(30.0, static_cast<double>(it->second.size()) * 1.4);
    return decayed + persistent;
}

void World::update_diplomatic_trust_and_pacts() {
    if (countries_.empty()) {
        return;
    }

    rebuild_country_index();

    for (const Country& country : countries_) {
        auto& history = betrayal_history_by_betrayer_[country.id];
        if (history.size() < country.betrayal_tick_log.size()) {
            history.insert(history.end(),
                           country.betrayal_tick_log.begin() + static_cast<std::ptrdiff_t>(history.size()),
                           country.betrayal_tick_log.end());
        }
    }

    for (Country& country : countries_) {
        for (auto& kv : country.trust_scores) {
            const uint16_t other_id = kv.first;
            const bool has_non_aggression = pallas::util::contains_id(country.has_non_aggression_with, other_id);
            const bool has_defense = pallas::util::contains_id(country.has_defense_pact_with, other_id);
            const bool has_trade = pallas::util::contains_id(country.has_trade_treaty_with, other_id);
            const bool is_embargoed = pallas::util::contains_id(country.embargoed_country_ids, other_id);

            const double trust_now = kv.second.to_double();
            double drift = (50.0 - trust_now) * 0.015;
            if (has_non_aggression) {
                drift += 0.11;
            }
            if (has_defense) {
                drift += 0.17;
            }
            if (has_trade) {
                drift += 0.06;
            }
            if (is_embargoed) {
                drift -= 0.25;
            }

            const double betrayal_penalty = betrayal_decay_score(other_id) * 0.08;
            kv.second = clamp_fixed(Fixed::from_double(trust_now + drift - betrayal_penalty),
                                    Fixed::from_int(0),
                                    Fixed::from_int(100));
        }
    }

    for (Country& country : countries_) {
        for (auto it = country.defense_pact_expiry_ticks.begin(); it != country.defense_pact_expiry_ticks.end(); ++it) {
            if (it->second > current_tick_) {
                continue;
            }
            const uint16_t other_id = it->first;
            Country* other = find_country(other_id);
            if (other == nullptr) {
                continue;
            }
            country.trust_scores[other_id] = clamp_fixed(country.trust_scores[other_id] - Fixed::from_double(1.0),
                                                         Fixed::from_int(0),
                                                         Fixed::from_int(100));
            other->trust_scores[country.id] = clamp_fixed(other->trust_scores[country.id] - Fixed::from_double(1.0),
                                                          Fixed::from_int(0),
                                                          Fixed::from_int(100));
            it->second = current_tick_ + 1;
            other->defense_pact_expiry_ticks[country.id] = current_tick_ + 1;
        }
        for (auto it = country.non_aggression_expiry_ticks.begin(); it != country.non_aggression_expiry_ticks.end(); ++it) {
            if (it->second > current_tick_) {
                continue;
            }
            const uint16_t other_id = it->first;
            Country* other = find_country(other_id);
            if (other == nullptr) {
                continue;
            }
            country.trust_scores[other_id] = clamp_fixed(country.trust_scores[other_id] - Fixed::from_double(0.6),
                                                         Fixed::from_int(0),
                                                         Fixed::from_int(100));
            other->trust_scores[country.id] = clamp_fixed(other->trust_scores[country.id] - Fixed::from_double(0.6),
                                                          Fixed::from_int(0),
                                                          Fixed::from_int(100));
            it->second = current_tick_ + 1;
            other->non_aggression_expiry_ticks[country.id] = current_tick_ + 1;
        }
    }

    for (size_t i = 0; i < countries_.size(); ++i) {
        for (size_t j = i + 1; j < countries_.size(); ++j) {
            enforce_pact_gates(&countries_[i], &countries_[j]);
        }
    }
}

void World::update_trade_and_internal_politics() {
    if (countries_.empty()) {
        return;
    }

    rebuild_country_index();

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

        const double supply = country.resource_food_reserves.to_double() + country.resource_oil_reserves.to_double() * 0.7 +
            country.resource_minerals_reserves.to_double() * 0.6 + country.resource_rare_earth_reserves.to_double() * 0.8;
        const double demand = country.gdp_output.to_double() * 0.85 + country.military_upkeep.to_double() * 0.9 + country.draft_level.to_double() * 0.35;
        const double shortage = std::max(0.0, demand - supply);
        const double embargo_intensity = std::min(5.0, static_cast<double>(country.embargoed_country_ids.size()));
        const double embargo_price_spike = 1.0 + embargo_intensity * 0.13 + shortage / 180.0;
        const double blended_price = country.import_price_index.to_double() * 0.68 + embargo_price_spike * 0.32;
        country.import_price_index = clamp_fixed(Fixed::from_double(blended_price), Fixed::from_double(0.70), Fixed::from_double(3.60));

        if (embargo_intensity > 0.0) {
            const double shock = std::max(0.0, country.import_price_index.to_double() - 1.0);
            country.economic_stability = clamp_fixed(country.economic_stability - Fixed::from_double(shock * 0.9), Fixed::from_int(0), Fixed::from_int(100));
            country.civilian_morale = clamp_fixed(country.civilian_morale - Fixed::from_double(shock * 0.75), Fixed::from_int(0), Fixed::from_int(100));
            country.politics.public_dissent = clamp_fixed(country.politics.public_dissent + Fixed::from_double(shock * 0.65), Fixed::from_int(0), Fixed::from_int(100));
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
            const double a_demand = a.gdp_output.to_double() * 0.9 + a.military_upkeep.to_double() * 0.8 + a.draft_level.to_double() * 0.3;
            const double b_demand = b.gdp_output.to_double() * 0.9 + b.military_upkeep.to_double() * 0.8 + b.draft_level.to_double() * 0.3;
            const double treaty_bonus = (pallas::util::contains_id(a.has_trade_treaty_with, b.id) &&
                                         pallas::util::contains_id(b.has_trade_treaty_with, a.id))
                ? 1.18
                : 1.0;
            const double trust_factor = (0.65 + std::min(0.35, (a.trust_scores[b.id].to_double() + b.trust_scores[a.id].to_double()) / 260.0)) * treaty_bonus;

            // Bilateral prices react to local shortages and embargo-induced import stress.
            const double a_price = std::clamp(0.75 + a.import_price_index.to_double() * 0.35 + std::max(0.0, a_demand - a_supply) / 140.0, 0.70, 3.80);
            const double b_price = std::clamp(0.75 + b.import_price_index.to_double() * 0.35 + std::max(0.0, b_demand - b_supply) / 140.0, 0.70, 3.80);

            const double a_export_capacity = std::max(0.0, a_supply - a_demand);
            const double b_export_capacity = std::max(0.0, b_supply - b_demand);
            const double a_import_need = std::max(0.0, a_demand - a_supply);
            const double b_import_need = std::max(0.0, b_demand - b_supply);

            const double a_to_b_volume = std::min(a_export_capacity, b_import_need) / 20.0;
            const double b_to_a_volume = std::min(b_export_capacity, a_import_need) / 20.0;
            double gross_flow = (a_to_b_volume * b_price - b_to_a_volume * a_price) * trust_factor;

            const double a_blockade = blockade_pressure_by_country_.count(a.id) == 0 ? 0.0 : blockade_pressure_by_country_.at(a.id);
            const double b_blockade = blockade_pressure_by_country_.count(b.id) == 0 ? 0.0 : blockade_pressure_by_country_.at(b.id);
            if (a_blockade > 0.0 || b_blockade > 0.0) {
                const bool maritime_route = !countries_share_land_frontier(a.id, b.id);
                if (maritime_route) {
                    const double route_risk = std::clamp(std::max(a_blockade, b_blockade), 0.0, 1.0);
                    gross_flow *= (1.0 - route_risk * 0.55);
                }
            }

            next_trade_balance[i] += Fixed::from_double(gross_flow);
            next_trade_balance[it->second] -= Fixed::from_double(gross_flow);

            const Fixed econ_bonus = Fixed::from_double(std::max(0.2, 0.45 + (a_to_b_volume + b_to_a_volume) * 0.24));
            a.economic_stability = clamp_fixed(a.economic_stability + econ_bonus, Fixed::from_int(0), Fixed::from_int(100));
            b.economic_stability = clamp_fixed(b.economic_stability + econ_bonus, Fixed::from_int(0), Fixed::from_int(100));
            a.civilian_morale = clamp_fixed(a.civilian_morale + econ_bonus / Fixed::from_int(3), Fixed::from_int(0), Fixed::from_int(100));
            b.civilian_morale = clamp_fixed(b.civilian_morale + econ_bonus / Fixed::from_int(3), Fixed::from_int(0), Fixed::from_int(100));

            a.import_price_index = clamp_fixed(Fixed::from_double(a.import_price_index.to_double() * 0.90 + a_price * 0.10), Fixed::from_double(0.70), Fixed::from_double(3.80));
            b.import_price_index = clamp_fixed(Fixed::from_double(b.import_price_index.to_double() * 0.90 + b_price * 0.10), Fixed::from_double(0.70), Fixed::from_double(3.80));
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
            c.import_price_index = clamp_fixed(c.import_price_index + Fixed::from_double(0.14), Fixed::from_double(0.70), Fixed::from_double(3.80));
        }

        c.trade_balance = clamp_fixed(next_trade_balance[i], Fixed::from_int(-100), Fixed::from_int(100));

        const double blockade_pressure = blockade_pressure_by_country_.count(c.id) == 0 ? 0.0 : blockade_pressure_by_country_.at(c.id);
        if (blockade_pressure > 0.0 && c.trade_balance > Fixed::from_int(0)) {
            const double penalty = std::clamp(0.40 + blockade_pressure * 0.30, 0.40, 0.70);
            c.trade_balance = clamp_fixed(Fixed::from_double(c.trade_balance.to_double() * (1.0 - penalty)), Fixed::from_int(-100), Fixed::from_int(100));
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(0.8 + penalty * 1.7), Fixed::from_int(0), Fixed::from_int(100));
            c.civilian_morale = clamp_fixed(c.civilian_morale - Fixed::from_double(0.4 + penalty * 1.1), Fixed::from_int(0), Fixed::from_int(100));
            c.import_price_index = clamp_fixed(c.import_price_index + Fixed::from_double(0.08 + penalty * 0.16), Fixed::from_double(0.70), Fixed::from_double(3.80));
        }

        const double strategic_bonus = strategic_economic_bonus_by_country_.count(c.id) == 0 ? 0.0 : strategic_economic_bonus_by_country_.at(c.id);
        if (strategic_bonus > 0.0) {
            c.economic_stability = clamp_fixed(c.economic_stability + Fixed::from_double(0.25 + strategic_bonus * 1.10), Fixed::from_int(0), Fixed::from_int(100));
            c.gdp_output = clamp_fixed(c.gdp_output + Fixed::from_double(strategic_bonus * 1.40), Fixed::from_int(0), Fixed::from_int(220));
            c.trade_balance = clamp_fixed(c.trade_balance + Fixed::from_double(strategic_bonus * 1.10), Fixed::from_int(-100), Fixed::from_int(100));
        }

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

        const double war_pressure = std::clamp((upkeep * 0.95 + draft * 0.75 + c.war_weariness.to_double() * 0.50 + c.war_duration_ticks * 0.42) / 120.0,
                                               0.0,
                                               1.8);
        const double desired_war_econ = std::clamp(war_pressure * 65.0, 0.0, 100.0);
        const double current_war_econ = c.war_economy_intensity.to_double();
        const double war_gap = desired_war_econ - current_war_econ;
        const double diminishing = 1.0 / (1.0 + std::abs(current_war_econ - 50.0) / 45.0);
        const double conversion_step = std::clamp(war_gap * 0.22 * diminishing, -3.4, 3.4);

        if (conversion_step >= 0.0) {
            // Mobilization empowers hawks and nationalists while economic liberals lose influence.
            const double mobilization_efficiency = std::clamp(1.0 - current_war_econ / 135.0, 0.25, 1.0);
            const double hawk_gain = conversion_step * mobilization_efficiency;
            c.faction_hawks = clamp_fixed(c.faction_hawks + Fixed::from_double(hawk_gain), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_nationalists = clamp_fixed(c.faction_nationalists + Fixed::from_double(hawk_gain * 0.45), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_populists = clamp_fixed(c.faction_populists - Fixed::from_double(hawk_gain * 0.35), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_economic_liberals = clamp_fixed(c.faction_economic_liberals - Fixed::from_double(hawk_gain * 0.55), Fixed::from_int(0), Fixed::from_int(100));
        } else {
            // Demobilization tends to restore liberal/populist influence with reconversion costs.
            const double reconversion = std::abs(conversion_step);
            c.faction_hawks = clamp_fixed(c.faction_hawks - Fixed::from_double(reconversion), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_populists = clamp_fixed(c.faction_populists + Fixed::from_double(reconversion * 0.45), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_economic_liberals = clamp_fixed(c.faction_economic_liberals + Fixed::from_double(reconversion * 0.45), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_nationalists = clamp_fixed(c.faction_nationalists + Fixed::from_double(reconversion * 0.10), Fixed::from_int(0), Fixed::from_int(100));
            c.industrial_capital = clamp_fixed(c.industrial_capital - Fixed::from_double(reconversion * 0.55), Fixed::from_int(1), Fixed::from_int(300));
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(reconversion * 0.35), Fixed::from_int(0), Fixed::from_int(100));
        }
        c.war_economy_intensity = clamp_fixed(c.war_economy_intensity + Fixed::from_double(conversion_step), Fixed::from_int(0), Fixed::from_int(100));

        if (c.war_economy_intensity > Fixed::from_int(55) && c.war_duration_ticks > 18) {
            const double decay = std::clamp((c.war_economy_intensity.to_double() - 55.0) * 0.035 + c.war_duration_ticks * 0.01, 0.0, 2.8);
            c.industrial_capital = clamp_fixed(c.industrial_capital - Fixed::from_double(decay), Fixed::from_int(1), Fixed::from_int(300));
            c.industrial_decay = clamp_fixed(c.industrial_decay + Fixed::from_double(decay * 0.6), Fixed::from_int(0), Fixed::from_int(100));
            c.technology_multiplier = clamp_fixed(c.technology_multiplier - Fixed::from_double(decay * 0.006), Fixed::from_double(0.35), Fixed::from_double(3.0));
        } else {
            c.industrial_decay = clamp_fixed(c.industrial_decay - Fixed::from_double(0.18), Fixed::from_int(0), Fixed::from_int(100));
        }

        const double trait_aggr = c.leader_traits.aggressive.to_double() / 100.0;
        const double trait_dipl = c.leader_traits.diplomatic.to_double() / 100.0;
        const double trait_corrupt = c.leader_traits.corrupt.to_double() / 100.0;
        const double trait_comp = c.leader_traits.competent.to_double() / 100.0;

        c.faction_hawks = clamp_fixed(c.faction_hawks + Fixed::from_double((trait_aggr - 0.50) * 0.9), Fixed::from_int(0), Fixed::from_int(100));
        c.faction_economic_liberals = clamp_fixed(c.faction_economic_liberals + Fixed::from_double((trait_comp - 0.45) * 0.6), Fixed::from_int(0), Fixed::from_int(100));
        c.faction_nationalists = clamp_fixed(c.faction_nationalists + Fixed::from_double((trait_aggr - trait_dipl) * 0.7), Fixed::from_int(0), Fixed::from_int(100));
        c.faction_populists = clamp_fixed(c.faction_populists + Fixed::from_double((trait_corrupt - 0.40) * 0.8), Fixed::from_int(0), Fixed::from_int(100));

        const double faction_sum = std::max(1.0,
            c.faction_hawks.to_double() + c.faction_economic_liberals.to_double() + c.faction_nationalists.to_double() + c.faction_populists.to_double());
        c.faction_hawks = Fixed::from_double(c.faction_hawks.to_double() * 100.0 / faction_sum);
        c.faction_economic_liberals = Fixed::from_double(c.faction_economic_liberals.to_double() * 100.0 / faction_sum);
        c.faction_nationalists = Fixed::from_double(c.faction_nationalists.to_double() * 100.0 / faction_sum);
        c.faction_populists = Fixed::from_double(c.faction_populists.to_double() * 100.0 / faction_sum);

        // Keep legacy faction channels aligned with the expanded four-faction model.
        c.faction_military = c.faction_hawks;
        c.faction_industrial = c.faction_economic_liberals;
        c.faction_civilian = c.faction_populists;

        const double estimated_revenue = std::max(6.0,
            c.gdp_output.to_double() * 0.92 + std::max(0.0, c.trade_balance.to_double()) * 0.46 + c.resource_reserve.to_double() * 0.08);
        const double estimated_spending = c.military_upkeep.to_double() * 1.08 + draft * 0.42 + c.war_economy_intensity.to_double() * 0.22 +
            (100.0 - c.economic_stability.to_double()) * 0.14;
        const double deficit = estimated_spending - estimated_revenue;

        if (deficit > 0.0) {
            const double debt_buildup = std::clamp((deficit / estimated_revenue) * 7.0, 0.0, 6.0);
            c.debt_to_gdp = clamp_fixed(c.debt_to_gdp + Fixed::from_double(debt_buildup), Fixed::from_int(0), Fixed::from_int(250));

            if (war_pressure > 0.55) {
                const double issuance = std::clamp(deficit * 0.16 + war_pressure * 2.0, 0.0, 5.5);
                c.war_bond_stock = clamp_fixed(c.war_bond_stock + Fixed::from_double(issuance), Fixed::from_int(0), Fixed::from_int(120));
                c.military_upkeep = clamp_fixed(c.military_upkeep + Fixed::from_double(issuance * 0.45), Fixed::from_int(0), Fixed::from_int(100));
                c.draft_level = clamp_fixed(c.draft_level + Fixed::from_double(issuance * 0.22), Fixed::from_int(0), Fixed::from_int(100));
                c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(issuance * 0.18), Fixed::from_int(0), Fixed::from_int(100));
            }
        } else {
            const double deleverage = std::clamp((-deficit / estimated_revenue) * 3.8, 0.0, 3.0);
            c.debt_to_gdp = clamp_fixed(c.debt_to_gdp - Fixed::from_double(deleverage), Fixed::from_int(0), Fixed::from_int(250));
        }

        const double debt_service = c.debt_to_gdp.to_double() * 0.028 + std::max(0.0, c.war_bond_stock.to_double() - 18.0) * 0.030;
        c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(debt_service * 0.26), Fixed::from_int(0), Fixed::from_int(100));
        c.politics.public_dissent = clamp_fixed(c.politics.public_dissent + Fixed::from_double(debt_service * 0.22), Fixed::from_int(0), Fixed::from_int(100));
        c.war_bond_stock = clamp_fixed(c.war_bond_stock - Fixed::from_double(0.42), Fixed::from_int(0), Fixed::from_int(120));

        double strongest_neighbor = 1.0;
        for (uint16_t neighbor_id : c.adjacent_country_ids) {
            const Country* neighbor = find_country(neighbor_id);
            if (neighbor == nullptr) {
                continue;
            }
            strongest_neighbor = std::max(strongest_neighbor, neighbor->military.weighted_total().to_double());
        }
        const double own_strength = std::max(1.0, c.military.weighted_total().to_double());
        const double perceived_threat = std::clamp((strongest_neighbor / own_strength) * 0.70 + c.escalation_level.to_double() / 8.0, 0.0, 1.0);
        const double weariness_delta = c.recent_combat_losses.to_double() * 0.09 + c.war_duration_ticks * 0.035 + draft * 0.018 +
            (1.0 - perceived_threat) * 1.10 - c.trade_partners.size() * 0.05 - c.civilian_morale.to_double() * 0.004;
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

        c.unrest_stage = compute_unrest_stage(c);
        if (c.unrest_stage == InternalUnrestStage::Protests) {
            c.civilian_morale = clamp_fixed(c.civilian_morale - Fixed::from_double(0.4), Fixed::from_int(0), Fixed::from_int(100));
            c.gdp_output = clamp_fixed(c.gdp_output - Fixed::from_double(0.3), Fixed::from_int(0), Fixed::from_int(200));
        } else if (c.unrest_stage == InternalUnrestStage::Strikes) {
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(1.1), Fixed::from_int(0), Fixed::from_int(100));
            c.industrial_output = clamp_fixed(c.industrial_output - Fixed::from_double(1.3), Fixed::from_int(0), Fixed::from_int(100));
            c.logistics_capacity = clamp_fixed(c.logistics_capacity - Fixed::from_double(0.8), Fixed::from_int(0), Fixed::from_int(100));
        } else if (c.unrest_stage == InternalUnrestStage::Riots) {
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(2.2), Fixed::from_int(0), Fixed::from_int(100));
            c.politics.government_stability = clamp_fixed(c.politics.government_stability - Fixed::from_double(2.0), Fixed::from_int(0), Fixed::from_int(100));
            c.supply_level = clamp_fixed(c.supply_level - Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_populists = clamp_fixed(c.faction_populists + Fixed::from_double(1.6), Fixed::from_int(0), Fixed::from_int(100));
        } else if (c.unrest_stage == InternalUnrestStage::CivilWar) {
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(4.0), Fixed::from_int(0), Fixed::from_int(100));
            c.politics.government_stability = clamp_fixed(c.politics.government_stability - Fixed::from_double(4.2), Fixed::from_int(0), Fixed::from_int(100));
            c.logistics_capacity = clamp_fixed(c.logistics_capacity - Fixed::from_double(2.0), Fixed::from_int(0), Fixed::from_int(100));
            c.military.units_infantry = clamp_fixed(c.military.units_infantry - Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(1000000));
            c.military.units_artillery = clamp_fixed(c.military.units_artillery - Fixed::from_double(0.4), Fixed::from_int(0), Fixed::from_int(1000000));
            c.reputation = clamp_fixed(c.reputation - Fixed::from_double(0.6), Fixed::from_int(0), Fixed::from_int(100));
        }

        if (c.war_weariness > Fixed::from_int(72)) {
            if (c.regime_type == RegimeType::Democratic) {
                c.diplomatic_stance = DiplomaticStance::Pacifist;
                c.draft_level = clamp_fixed(c.draft_level - Fixed::from_double(2.0), Fixed::from_int(0), Fixed::from_int(100));
            } else if (c.regime_type == RegimeType::Authoritarian) {
                c.politics.public_dissent = clamp_fixed(c.politics.public_dissent - Fixed::from_double(1.6), Fixed::from_int(0), Fixed::from_int(100));
                c.war_weariness = clamp_fixed(c.war_weariness - Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
                c.reputation = clamp_fixed(c.reputation - Fixed::from_double(0.9), Fixed::from_int(0), Fixed::from_int(100));
            }
        }

        const double coup_score = c.faction_hawks.to_double() * 0.30 + c.faction_nationalists.to_double() * 0.16 + c.politics.public_dissent.to_double() * 0.26 +
            (100.0 - c.politics.government_stability.to_double()) * 0.24 + (100.0 - c.economic_stability.to_double()) * 0.16 -
            c.faction_populists.to_double() * 0.14 + c.debt_to_gdp.to_double() * 0.17 + c.war_bond_stock.to_double() * 0.06;
        c.coup_risk = clamp_fixed(Fixed::from_double(coup_score / 0.9), Fixed::from_int(0), Fixed::from_int(100));

        c.leader_tenure_ticks += 1;

        c.election_cycle -= 1;
        if (c.election_cycle <= 0) {
            const double performance = c.civilian_morale.to_double() + c.economic_stability.to_double() + c.trade_balance.to_double() * 0.4 - c.war_weariness.to_double();
            const int dominant_faction = dominant_faction_index(c);
            if (performance < 85.0 && c.faction_hawks > c.faction_populists) {
                c.diplomatic_stance = DiplomaticStance::Aggressive;
                c.politics.government_stability = clamp_fixed(c.politics.government_stability - Fixed::from_double(2.5), Fixed::from_int(0), Fixed::from_int(100));
                c.faction_hawks = clamp_fixed(c.faction_hawks + Fixed::from_double(4.0), Fixed::from_int(0), Fixed::from_int(100));
            } else if (performance > 150.0 || c.faction_populists >= c.faction_hawks) {
                c.diplomatic_stance = DiplomaticStance::Pacifist;
                c.politics.government_stability = clamp_fixed(c.politics.government_stability + Fixed::from_double(4.0), Fixed::from_int(0), Fixed::from_int(100));
                c.faction_populists = clamp_fixed(c.faction_populists + Fixed::from_double(3.5), Fixed::from_int(0), Fixed::from_int(100));
            } else {
                c.diplomatic_stance = DiplomaticStance::Neutral;
                c.politics.government_stability = clamp_fixed(c.politics.government_stability + Fixed::from_double(1.2), Fixed::from_int(0), Fixed::from_int(100));
                c.faction_economic_liberals = clamp_fixed(c.faction_economic_liberals + Fixed::from_double(2.0), Fixed::from_int(0), Fixed::from_int(100));
            }
            reroll_leader_traits(&c, base_seed_ + c.id * 31ULL, current_tick_ + 0xA11EULL, dominant_faction);
            c.election_cycle = 10 + static_cast<int>((mix_u64(base_seed_ + c.id + current_tick_) % 9ULL));
            c.politics.public_dissent = clamp_fixed(c.politics.public_dissent - Fixed::from_double(3.0), Fixed::from_int(0), Fixed::from_int(100));
        }

        const double coup_roll = normalized_event_roll(base_seed_ + c.id, current_tick_ + c.id * 17ULL);
        if (c.coup_risk > Fixed::from_int(72) && c.faction_hawks > Fixed::from_int(42) && coup_roll < c.coup_risk.to_double() / 180.0) {
            c.diplomatic_stance = DiplomaticStance::Aggressive;
            c.politics.government_stability = clamp_fixed(Fixed::from_double(52.0 - coup_roll * 10.0), Fixed::from_int(0), Fixed::from_int(100));
            c.civilian_morale = clamp_fixed(c.civilian_morale - Fixed::from_double(9.0), Fixed::from_int(0), Fixed::from_int(100));
            c.economic_stability = clamp_fixed(c.economic_stability - Fixed::from_double(6.0), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_hawks = clamp_fixed(c.faction_hawks + Fixed::from_double(10.0), Fixed::from_int(0), Fixed::from_int(100));
            c.faction_populists = clamp_fixed(c.faction_populists - Fixed::from_double(8.0), Fixed::from_int(0), Fixed::from_int(100));
            c.trade_partners.clear();
            c.embargoed_country_ids.clear();
            c.election_cycle = 18;
            c.regime_type = RegimeType::Authoritarian;
            reroll_leader_traits(&c, base_seed_ + c.id * 43ULL, current_tick_ + 0xC0A0ULL, 0);
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
    auto& history = betrayal_history_by_betrayer_[country.id];
    if (history.size() < country.betrayal_tick_log.size()) {
        history.insert(history.end(),
                       country.betrayal_tick_log.begin() + static_cast<std::ptrdiff_t>(history.size()),
                       country.betrayal_tick_log.end());
    }
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
            const double betrayal_bonus = std::min(22.0, betrayal_decay_score(neighbor_id) * 1.2);
            const double updated_conf = std::clamp(prior_conf * 0.74 + (18.0 + visibility * 78.0 + betrayal_bonus) * 0.26,
                                                   5.0,
                                                   100.0);
            observer.opponent_model_confidence[neighbor_id] = Fixed::from_double(updated_conf);

            mean_confidence += updated_conf;
            ++confidence_count;
            threat_pressure += std::max(0.0, actual_strength - own_strength) / own_strength;
        }

        const double avg_confidence = confidence_count == 0 ? 25.0 : mean_confidence / static_cast<double>(confidence_count);
        const double betrayal_pressure = std::min(2.8, betrayal_decay_score(observer.id) * 0.08);
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
    std::vector<Fixed> next_industrial_capital(countries_.size());
    std::vector<Fixed> next_labor_participation(countries_.size());
    std::vector<Fixed> next_technology_multiplier(countries_.size());
    std::vector<Fixed> next_gdp(countries_.size());
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
        const Fixed industry_boost = country.gdp_output / Fixed::from_int(150);
        const Fixed logistics_boost = country.logistics_capacity / Fixed::from_int(180);
        const Fixed reserve_boost = country.resource_reserve / Fixed::from_int(220);
        const Fixed politics_drag = country.politics.public_dissent / Fixed::from_int(140);
        const Fixed upkeep_drag = country.military_upkeep / Fixed::from_int(170);
        const Fixed weariness_drag = country.war_weariness / Fixed::from_int(150);
        const Fixed debt_drag = country.debt_to_gdp / Fixed::from_int(320);
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
        const Fixed armor_regen = clamp_fixed((country.gdp_output / Fixed::from_int(190)) + (country.resources.oil / Fixed::from_int(260)) - weariness_drag,
                              Fixed::from_int(0), Fixed::from_double(0.9));
        const Fixed artillery_regen = clamp_fixed((country.gdp_output / Fixed::from_int(210)) + (country.resources.minerals / Fixed::from_int(280)) - (border_pressure / Fixed::from_int(40)),
                              Fixed::from_int(0), Fixed::from_double(0.8));
        const Fixed fighter_regen = clamp_fixed((country.technology_level / Fixed::from_int(220)) + (country.logistics_capacity / Fixed::from_int(260)) - (country.weather_severity / Fixed::from_int(320)),
                            Fixed::from_int(0), Fixed::from_double(0.55));
        const Fixed bomber_regen = clamp_fixed((country.technology.drone_operations / Fixed::from_int(240)) + (country.resources.rare_earth / Fixed::from_int(320)) - weariness_drag,
                               Fixed::from_int(0), Fixed::from_double(0.35));
        const Fixed surface_regen = clamp_fixed((country.resources.oil / Fixed::from_int(320)) + (country.gdp_output / Fixed::from_int(320)) - (border_pressure / Fixed::from_int(55)),
                            Fixed::from_int(0), Fixed::from_double(0.28));
        const Fixed submarine_regen = clamp_fixed((country.intelligence_level / Fixed::from_int(320)) + (country.resources.rare_earth / Fixed::from_int(360)) - (border_pressure / Fixed::from_int(60)),
                              Fixed::from_int(0), Fixed::from_double(0.24));

        next_morale[static_cast<size_t>(i)] = clamp_fixed(country.civilian_morale + stance_morale_modifier - morale_decay - weariness_drag, Fixed::from_int(0), Fixed::from_int(100));
        next_econ[static_cast<size_t>(i)] = clamp_fixed(country.economic_stability + stance_econ_modifier + trust_bonus - econ_decay - politics_drag - upkeep_drag - debt_drag,
                                Fixed::from_int(0),
                                Fixed::from_int(100));
        next_infantry[static_cast<size_t>(i)] = clamp_fixed(country.military.units_infantry + infantry_regen - (border_pressure / Fixed::from_int(10)) - (country.weather_severity / Fixed::from_int(280)) + (country.draft_level / Fixed::from_int(180)), Fixed::from_int(0), Fixed::from_int(1000000));
        next_armor[static_cast<size_t>(i)] = clamp_fixed(country.military.units_armor + armor_regen - (border_pressure / Fixed::from_int(18)) - (country.weather_severity / Fixed::from_int(360)), Fixed::from_int(0), Fixed::from_int(1000000));
        next_artillery[static_cast<size_t>(i)] = clamp_fixed(country.military.units_artillery + artillery_regen - (border_pressure / Fixed::from_int(16)), Fixed::from_int(0), Fixed::from_int(1000000));
        next_fighters[static_cast<size_t>(i)] = clamp_fixed(country.military.units_air_fighter + fighter_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_bombers[static_cast<size_t>(i)] = clamp_fixed(country.military.units_air_bomber + bomber_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_surface[static_cast<size_t>(i)] = clamp_fixed(country.military.units_naval_surface + surface_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_submarines[static_cast<size_t>(i)] = clamp_fixed(country.military.units_naval_submarine + submarine_regen, Fixed::from_int(0), Fixed::from_int(1000000));
        next_logistics[static_cast<size_t>(i)] = clamp_fixed(country.logistics_capacity + (country.economic_stability / Fixed::from_int(260)) - (border_pressure / Fixed::from_int(30)) - weariness_drag - (country.import_price_index / Fixed::from_int(20)),
                                                             Fixed::from_int(0), Fixed::from_int(100));
        next_intel[static_cast<size_t>(i)] = clamp_fixed(country.intelligence_level + (country.technology_level / Fixed::from_int(300)) - (border_pressure / Fixed::from_int(35)),
                                                         Fixed::from_int(0), Fixed::from_int(100));

        const double war_drag = country.war_economy_intensity.to_double() / 320.0 + country.industrial_decay.to_double() / 140.0;
        const double capital_next = std::clamp(
            country.industrial_capital.to_double() + country.gdp_output.to_double() * 0.018 + country.faction_industrial.to_double() * 0.012 - war_drag,
            1.0,
            320.0);
        const double labor_next = std::clamp(
            country.labor_participation.to_double() * 0.88 + (0.35 + country.civilian_morale.to_double() / 170.0 - country.draft_level.to_double() / 240.0) * 0.12,
            0.25,
            0.92);
        const double tech_multiplier_next = std::clamp(
            country.technology_multiplier.to_double() * 0.94 + (0.50 + country.technology_level.to_double() / 95.0 - country.politics.corruption.to_double() / 260.0) * 0.06,
            0.35,
            3.20);
        const double labor_units = std::max(0.2, (static_cast<double>(country.population) * labor_next) / 1'000'000.0);
        const double alpha = std::clamp(0.26 + country.faction_industrial.to_double() / 420.0 + country.war_economy_intensity.to_double() / 620.0, 0.20, 0.62);
        const double gdp_from_factors = cobb_douglas_output(tech_multiplier_next, capital_next, labor_units, alpha);
        const double gdp_next = std::clamp(gdp_from_factors * 8.2, 2.0, 220.0);
        const double industry_index = std::clamp(gdp_next * 0.52 + country.economic_stability.to_double() * 0.24 - border_pressure.to_double() * 0.16, 0.0, 100.0);

        next_industrial_capital[static_cast<size_t>(i)] = Fixed::from_double(capital_next);
        next_labor_participation[static_cast<size_t>(i)] = Fixed::from_double(labor_next);
        next_technology_multiplier[static_cast<size_t>(i)] = Fixed::from_double(tech_multiplier_next);
        next_gdp[static_cast<size_t>(i)] = Fixed::from_double(gdp_next);
        next_industry[static_cast<size_t>(i)] = Fixed::from_double(industry_index);

        next_technology[static_cast<size_t>(i)] = clamp_fixed(country.technology_level + (country.intelligence_level / Fixed::from_int(320)) - (border_pressure / Fixed::from_int(40)),
                                                              Fixed::from_int(0), Fixed::from_int(100));
        next_reserve[static_cast<size_t>(i)] = clamp_fixed(country.resource_reserve + (Fixed::from_double(gdp_next) / Fixed::from_int(300)) - (border_pressure / Fixed::from_int(22)),
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
        countries_[i].industrial_capital = next_industrial_capital[i];
        countries_[i].labor_participation = next_labor_participation[i];
        countries_[i].technology_multiplier = next_technology_multiplier[i];
        countries_[i].gdp_output = next_gdp[i];
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