#include "model.h"
#include "battle_runtime.h"
#include "simulation_engine.h"
#include "tensor.h"
#include "test_framework.h"
#include "train_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

class TempPath {
public:
    explicit TempPath(const std::string& suffix) {
        char tmpl[] = "/tmp/pallas_test_XXXXXX";
        const int fd = mkstemp(tmpl);
        if (fd >= 0) {
            close(fd);
        }
        path_ = std::string(tmpl) + suffix;
        std::error_code ec;
        std::filesystem::rename(tmpl, path_, ec);
        if (ec) {
            path_ = std::string(tmpl);
        }
    }

    ~TempPath() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

sim::Country make_country(uint16_t id,
                          const char* name,
                          int64_t army,
                          int64_t navy,
                          int64_t air,
                          int64_t missiles,
                          sim::DiplomaticStance stance,
                          std::vector<uint16_t> adjacency) {
    sim::Country country;
    country.id = id;
    country.name = name;
    country.population = 1000000 + id * 100000;
    country.capital = {static_cast<int32_t>(id), static_cast<int32_t>(id)};
    country.shape = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    country.military.units_infantry = sim::Fixed::from_int(army);
    country.military.units_armor = sim::Fixed::from_int(std::max<int64_t>(8, army / 5));
    country.military.units_artillery = sim::Fixed::from_int(std::max<int64_t>(6, army / 7));
    country.military.units_naval_surface = sim::Fixed::from_int(navy);
    country.military.units_naval_submarine = sim::Fixed::from_int(std::max<int64_t>(2, navy / 3));
    country.military.units_air_fighter = sim::Fixed::from_int(std::max<int64_t>(4, air / 2));
    country.military.units_air_bomber = sim::Fixed::from_int(std::max<int64_t>(3, air / 3 + missiles / 4));
    country.civilian_morale = sim::Fixed::from_int(80);
    country.economic_stability = sim::Fixed::from_int(75);
    country.diplomatic_stance = stance;
    country.adjacent_country_ids = std::move(adjacency);
    return country;
}

TEST_CASE(test_forward_backward_shapes) {
    ModelConfig cfg;
    cfg.hidden_layers = {64, 32};
    cfg.activation = "relu";
    cfg.norm = "layernorm";
    cfg.use_dropout = false;

    Model model(16, 16, cfg);
    model.set_training(true);

    Tensor x({1, 16}, 0.0f);
    x.data[3] = 1.0f;
    Tensor out = model.forward(x);
    EXPECT_TRUE(out.shape.size() == 2 && out.shape[0] == 1 && out.shape[1] == 16, "forward output shape");

    Tensor grad = grad_cross_entropy_advanced(out, 2, 0.0f, 1.0f);
    model.zero_grad();
    model.backward(grad);
    EXPECT_TRUE(model.gradient_size() > 0, "gradient size non-zero");
}

TEST_CASE(test_state_roundtrip) {
    ModelConfig cfg;
    cfg.hidden_layers = {32};
    cfg.activation = "tanh";
    cfg.norm = "none";
    cfg.use_dropout = false;

    OptimizerConfig opt;
    opt.type = "adam";

    Model src(8, 8, cfg);
    src.configure_optimizer(opt);
    src.set_training(true);

    Tensor x({1, 8}, 0.0f);
    x.data[1] = 1.0f;
    const Tensor src_out = src.forward(x);

    ModelTrainingMetadata md;
    md.timestamp_unix = 1;
    md.epoch = 2;
    md.val_loss = 3.0f;
    md.val_top1 = 0.25f;
    md.optimizer = "adam";

    TempPath path(".state.bin");
    src.save_state(path.path(), 8, 8, cfg, md);

    Model dst(8, 8, cfg);
    dst.configure_optimizer(opt);
    ModelFileInfo info;
    const bool loaded = dst.load_state(path.path(), 8, 8, &info);
    EXPECT_TRUE(loaded, "load_state roundtrip success");
    EXPECT_TRUE(info.metadata.epoch == 2, "metadata epoch preserved");

    const Tensor dst_out = dst.forward(x);
    EXPECT_TRUE(dst_out.data.size() == src_out.data.size(), "roundtrip output size");

    float max_diff = 0.0f;
    for (size_t i = 0; i < src_out.data.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(src_out.data[i] - dst_out.data[i]));
    }
    EXPECT_TRUE(max_diff < 1e-5f, "roundtrip output equality");
}

TEST_CASE(test_train_config_validation) {
    TrainConfig cfg;
    std::string err;
    EXPECT_TRUE(validate_train_config(cfg, err), "default train config valid");
    cfg.batch_size = 0;
    EXPECT_TRUE(!validate_train_config(cfg, err), "invalid batch size rejected");
}

TEST_CASE(test_grid_binary_roundtrip) {
    sim::GridMap original(4, 3);
    for (uint32_t y = 0; y < original.height(); ++y) {
        for (uint32_t x = 0; x < original.width(); ++x) {
            original.set(x, y, static_cast<uint16_t>((x + y) % 3));
        }
    }

    TempPath path(".map.bin");
    EXPECT_TRUE(original.save_binary(path.path()), "map save binary");

    sim::GridMap loaded;
    EXPECT_TRUE(loaded.load_binary(path.path()), "map load binary");
    EXPECT_TRUE(loaded.width() == original.width() && loaded.height() == original.height(), "map dimensions preserved");
    EXPECT_TRUE(loaded.flattened_country_ids() == original.flattened_country_ids(), "map cell data preserved");
}

TEST_CASE(test_world_tick_and_determinism) {
    sim::GridMap map(6, 2);
    for (uint32_t y = 0; y < map.height(); ++y) {
        for (uint32_t x = 0; x < map.width(); ++x) {
            map.set(x, y, x < 3 ? 1 : 2);
        }
    }

    sim::World world_a(42, 3600);
    sim::World world_b(42, 3600);

    world_a.set_map(map);
    world_b.set_map(map);

    world_a.add_country(make_country(1, "Alpha", 120, 30, 20, 10, sim::DiplomaticStance::Aggressive, {2}));
    world_a.add_country(make_country(2, "Beta", 130, 28, 22, 8, sim::DiplomaticStance::Neutral, {1}));

    world_b.add_country(make_country(1, "Alpha", 120, 30, 20, 10, sim::DiplomaticStance::Aggressive, {2}));
    world_b.add_country(make_country(2, "Beta", 130, 28, 22, 8, sim::DiplomaticStance::Neutral, {1}));

    world_a.schedule_event(std::make_unique<sim::AttackEvent>(1, 2, sim::Fixed::from_double(0.95), sim::Fixed::from_double(1.10)), 0);
    world_a.schedule_event(std::make_unique<sim::NegotiationEvent>(1, 2), 1);

    world_b.schedule_event(std::make_unique<sim::AttackEvent>(1, 2, sim::Fixed::from_double(0.95), sim::Fixed::from_double(1.10)), 0);
    world_b.schedule_event(std::make_unique<sim::NegotiationEvent>(1, 2), 1);

    world_a.run_ticks(4);
    world_b.run_ticks(4);

    EXPECT_TRUE(world_a.current_tick() == 4 && world_b.current_tick() == 4, "fixed-step tick cadence");
    EXPECT_TRUE(world_a.tick_seconds() == 3600, "tick duration is one hour");

    const auto& a_countries = world_a.countries();
    const auto& b_countries = world_b.countries();
    EXPECT_TRUE(a_countries.size() == b_countries.size(), "deterministic country count");

    for (size_t i = 0; i < a_countries.size(); ++i) {
        EXPECT_TRUE(a_countries[i].military.units_infantry.raw() == b_countries[i].military.units_infantry.raw(), "deterministic infantry state");
        EXPECT_TRUE(a_countries[i].civilian_morale.raw() == b_countries[i].civilian_morale.raw(), "deterministic morale state");
        EXPECT_TRUE(a_countries[i].economic_stability.raw() == b_countries[i].economic_stability.raw(), "deterministic economic state");
    }
    EXPECT_TRUE(world_a.map().flattened_country_ids() == world_b.map().flattened_country_ids(), "deterministic territorial map");
    EXPECT_TRUE(world_a.random_seed_log() == world_b.random_seed_log(), "random seed log reproducible");
}

TEST_CASE(test_seed_changes_stochastic_outcome) {
    sim::GridMap map(4, 2);
    for (uint32_t y = 0; y < map.height(); ++y) {
        for (uint32_t x = 0; x < map.width(); ++x) {
            map.set(x, y, x < 2 ? 1 : 2);
        }
    }

    sim::World world_a(7, 3600);
    sim::World world_b(999, 3600);

    world_a.set_map(map);
    world_b.set_map(map);

    world_a.add_country(make_country(1, "Alpha", 150, 30, 20, 10, sim::DiplomaticStance::Aggressive, {2}));
    world_a.add_country(make_country(2, "Beta", 160, 28, 22, 8, sim::DiplomaticStance::Neutral, {1}));

    world_b.add_country(make_country(1, "Alpha", 150, 30, 20, 10, sim::DiplomaticStance::Aggressive, {2}));
    world_b.add_country(make_country(2, "Beta", 160, 28, 22, 8, sim::DiplomaticStance::Neutral, {1}));

    world_a.schedule_event(std::make_unique<sim::AttackEvent>(1, 2, sim::Fixed::from_double(1.00), sim::Fixed::from_double(1.05)), 0);
    world_b.schedule_event(std::make_unique<sim::AttackEvent>(1, 2, sim::Fixed::from_double(1.00), sim::Fixed::from_double(1.05)), 0);

    world_a.run_tick();
    world_b.run_tick();

    EXPECT_TRUE(!world_a.random_seed_log().empty() && !world_b.random_seed_log().empty(), "seed logs captured");
    EXPECT_TRUE(world_a.random_seed_log()[0] != world_b.random_seed_log()[0], "different base seeds produce different event seeds");
}

TEST_CASE(test_model_decide_interface) {
    ModelConfig cfg;
    cfg.hidden_layers = {8};
    cfg.activation = "relu";
    cfg.norm = "none";
    cfg.use_dropout = false;

    Model model(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, cfg);

    WorldSnapshot snapshot;
    snapshot.tick = 7;

    CountrySnapshot self;
    self.id = 10;
    self.units_infantry_milli = 220000;
    self.units_armor_milli = 50000;
    self.units_artillery_milli = 36000;
    self.units_air_fighter_milli = 24000;
    self.units_air_bomber_milli = 16000;
    self.units_naval_surface_milli = 18000;
    self.units_naval_submarine_milli = 9000;
    self.supply_level_milli = 76000;
    self.supply_capacity_milli = 80000;
    self.reputation_milli = 62000;
    self.economic_stability_milli = 70000;
    self.civilian_morale_milli = 72000;
    self.diplomatic_stance = 0;
    self.adjacent_country_ids = {11};
    snapshot.countries.push_back(self);

    CountrySnapshot enemy;
    enemy.id = 11;
    enemy.units_infantry_milli = 180000;
    enemy.units_armor_milli = 42000;
    enemy.units_artillery_milli = 30000;
    enemy.units_air_fighter_milli = 21000;
    enemy.units_air_bomber_milli = 13000;
    enemy.units_naval_surface_milli = 14000;
    enemy.units_naval_submarine_milli = 7000;
    enemy.supply_level_milli = 62000;
    enemy.supply_capacity_milli = 70000;
    enemy.reputation_milli = 54000;
    enemy.economic_stability_milli = 60000;
    enemy.civilian_morale_milli = 65000;
    enemy.diplomatic_stance = 1;
    snapshot.countries.push_back(enemy);

    const ModelDecision decision = model.decide(snapshot, 10);
    EXPECT_TRUE(decision.actor_country_id == 10, "decide actor id preserved");
    EXPECT_TRUE(static_cast<uint32_t>(decision.strategy) < battle_common::kBattleOutputDim,
                "decide strategy returns valid tactical action");
}

TEST_CASE(test_model_dimension_accessors) {
    ModelConfig cfg;
    cfg.hidden_layers = {12, 6};
    cfg.activation = "relu";
    cfg.norm = "none";
    cfg.use_dropout = false;

    Model model(16, 10, cfg);
    EXPECT_TRUE(model.input_dim() == 16, "model input_dim accessor matches constructor");
    EXPECT_TRUE(model.output_dim() == 10, "model output_dim accessor matches constructor");
}

TEST_CASE(test_replay_roundtrip) {
    sim::World world(123, 3600);
    sim::GridMap map(4, 1);
    map.set(0, 0, 1);
    map.set(1, 0, 1);
    map.set(2, 0, 2);
    map.set(3, 0, 2);
    world.set_map(map);

    world.add_country(make_country(1, "Alpha", 120, 20, 20, 10, sim::DiplomaticStance::Aggressive, {2}));
    world.add_country(make_country(2, "Beta", 115, 18, 18, 8, sim::DiplomaticStance::Neutral, {1}));
    world.mutable_countries()[0].trade_balance = sim::Fixed::from_int(12);
    world.mutable_countries()[0].trade_partners = {2};
    world.mutable_countries()[0].resource_oil_reserves = sim::Fixed::from_int(64);
    world.mutable_countries()[0].coup_risk = sim::Fixed::from_int(9);

    ModelConfig cfg;
    cfg.hidden_layers = {8};
    cfg.activation = "relu";
    cfg.norm = "none";
    cfg.use_dropout = false;

    battle::ModelManager manager;
    manager.add_model({"test_model", "red", std::make_shared<Model>(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, cfg), {1}});

    const std::vector<battle::DecisionEnvelope> decisions = manager.gather_decisions(world);
    manager.apply_decisions(world, decisions);
    world.run_tick();

    TempPath path(".replay.bin");
    battle::ReplayLogger logger;
    EXPECT_TRUE(logger.open(path.path()), "replay logger open");
    EXPECT_TRUE(logger.write_tick(world, manager, decisions), "replay logger write tick");
    logger.close();

    battle::ReplayReader reader;
    EXPECT_TRUE(reader.open(path.path()), "replay reader open");
    battle::ReplayFrame frame;
    EXPECT_TRUE(reader.read_next(&frame), "replay read first frame");
    EXPECT_TRUE(frame.countries.size() == 2, "replay country count");
    EXPECT_TRUE(!frame.decisions.empty(), "replay decisions captured");
    EXPECT_TRUE(frame.countries[0].trade_balance_milli == world.countries()[0].trade_balance.raw(),
                "replay preserves trade balance");
    EXPECT_TRUE(frame.countries[0].trade_partner_ids == world.countries()[0].trade_partners,
                "replay preserves trade partners");
}

TEST_CASE(test_replace_model_weights_dimension_rejection) {
    ModelConfig battle_cfg;
    battle_cfg.hidden_layers = {8};
    battle_cfg.activation = "relu";
    battle_cfg.norm = "none";
    battle_cfg.use_dropout = false;

    ModelConfig wrong_cfg;
    wrong_cfg.hidden_layers = {12};
    wrong_cfg.activation = "relu";
    wrong_cfg.norm = "none";
    wrong_cfg.use_dropout = false;

    TempPath wrong_path(".wrong_model.bin");
    ModelTrainingMetadata md;
    md.timestamp_unix = 1;
    md.epoch = 1;
    md.val_loss = 0.1f;
    md.val_top1 = 0.1f;
    md.optimizer = "adam";

    Model wrong_model(16, 16, wrong_cfg);
    wrong_model.save_state(wrong_path.path(), 16, 16, wrong_cfg, md);

    battle::ModelManager manager;
    manager.add_model({"slot_a", "red", std::make_shared<Model>(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, battle_cfg), {1}});

    std::string error;
    const bool replaced = manager.replace_model_weights("slot_a", wrong_path.path(), &error);
    EXPECT_TRUE(!replaced, "replace_model_weights rejects architecture mismatch");
    EXPECT_TRUE(!error.empty(), "replace_model_weights mismatch returns error");
}

TEST_CASE(test_team_target_selection) {
    ModelConfig cfg;
    cfg.hidden_layers = {8};
    cfg.activation = "relu";
    cfg.norm = "none";
    cfg.use_dropout = false;

    battle::ModelManager manager;
    manager.add_model({"red_strat_ai", "red", std::make_shared<Model>(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, cfg), {1, 2}});
    manager.add_model({"blue_strat_ai", "blue", std::make_shared<Model>(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, cfg), {3}});

    const std::vector<std::string> red_slots = manager.model_slots_for_team("red");
    EXPECT_TRUE(red_slots.size() == 1 && red_slots[0] == "red_strat_ai", "model_slots_for_team lists all red slots");

    const std::vector<std::string> blue_slots = manager.model_slots_for_team("blue");
    EXPECT_TRUE(blue_slots.size() == 1 && blue_slots[0] == "blue_strat_ai", "model_slots_for_team lists all blue slots");

    const std::vector<std::string> unknown_slots = manager.model_slots_for_team("green");
    EXPECT_TRUE(unknown_slots.empty(), "unknown team has no slots");
}

}  // namespace

int main() {
    return pallas_test::run_all();
}
