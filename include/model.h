#ifndef MODEL_H
#define MODEL_H

#include "battle_common.h"
#include "linear.h"
#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct ModelTrainingMetadata {
    uint64_t timestamp_unix = 0;
    size_t epoch = 0;
    float val_loss = 0.0f;
    float val_top1 = 0.0f;
    std::string optimizer;
};

struct ModelFileInfo {
    uint32_t version = 0;
    uint64_t architecture_hash = 0;
    ModelTrainingMetadata metadata;
};

struct ModelConfig {
    std::vector<size_t> hidden_layers;
    std::string activation = "relu";
    std::string norm = "layernorm";
    bool use_dropout = true;
    float dropout_prob = 0.1f;
    float leaky_relu_alpha = 0.01f;
};

struct OptimizerConfig {
    std::string type = "adam";
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.0f;
};

enum class Strategy : uint8_t {
    Attack = 0,
    Defend = 1,
    Negotiate = 2,
    Surrender = 3,
    TransferWeapons = 4,
    FocusEconomy = 5,
    DevelopTechnology = 6,
    FormAlliance = 7,
    Betray = 8,
    CyberOperation = 9,
    SignTradeAgreement = 10,
    CancelTradeAgreement = 11,
    ImposeEmbargo = 12,
    InvestInResourceExtraction = 13,
    ReduceMilitaryUpkeep = 14,
    SuppressDissent = 15,
    HoldElections = 16,
    CoupAttempt = 17,
    ProposeDefensePact = 18,
    ProposeNonAggression = 19,
    BreakTreaty = 20,
    RequestIntel = 21,
    DeployUnits = 22,
    TacticalNuke = 23,
    StrategicNuke = 24,
    CyberAttack = 25,
};

struct NegotiationTerms {
    std::string type = "ceasefire";
    std::string details;
};

struct CountrySnapshot {
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
    int64_t technology_milli = 0;
    int64_t resource_reserve_milli = 0;
    int64_t supply_level_milli = 0;
    int64_t supply_capacity_milli = 0;
    int64_t weather_severity_milli = 0;
    int64_t seasonal_effect_milli = 0;
    int64_t supply_stockpile_milli = 0;
    int64_t terrain_mountains_milli = 0;
    int64_t terrain_forests_milli = 0;
    int64_t terrain_urban_milli = 0;
    int64_t tech_missile_defense_milli = 0;
    int64_t tech_cyber_warfare_milli = 0;
    int64_t tech_electronic_warfare_milli = 0;
    int64_t tech_drone_ops_milli = 0;
    int64_t resource_oil_milli = 0;
    int64_t resource_minerals_milli = 0;
    int64_t resource_food_milli = 0;
    int64_t resource_rare_earth_milli = 0;
    int64_t trade_balance_milli = 0;
    std::vector<uint16_t> trade_partner_ids;
    int64_t resource_oil_reserves_milli = 0;
    int64_t resource_minerals_reserves_milli = 0;
    int64_t resource_food_reserves_milli = 0;
    int64_t resource_rare_earth_reserves_milli = 0;
    int64_t military_upkeep_milli = 0;
    int64_t faction_military_milli = 0;
    int64_t faction_industrial_milli = 0;
    int64_t faction_civilian_milli = 0;
    int64_t coup_risk_milli = 0;
    int32_t election_cycle = 0;
    int64_t draft_level_milli = 0;
    int64_t war_weariness_milli = 0;
    int64_t gov_stability_milli = 0;
    int64_t public_dissent_milli = 0;
    int64_t corruption_milli = 0;
    int64_t nuclear_readiness_milli = 0;
    int64_t deterrence_posture_milli = 0;
    int64_t reputation_milli = 0;
    int64_t escalation_level_milli = 0;
    int64_t trust_average_milli = 50000;
    std::unordered_map<uint16_t, int64_t> trust_in_milli;
    std::unordered_map<uint16_t, int64_t> believed_army_size_milli;
    int32_t recent_betrayals = 0;
    int64_t strategic_depth_milli = 2000;
    std::unordered_map<uint16_t, int64_t> opponent_model_confidence_milli;
    uint8_t diplomatic_stance = 1;
    bool second_strike_capable = false;
    std::vector<uint16_t> adjacent_country_ids;
    std::vector<uint16_t> allied_country_ids;
    std::vector<uint16_t> defense_pact_ids;
    std::vector<uint16_t> non_aggression_pact_ids;
    std::vector<uint16_t> trade_treaty_ids;
    std::unordered_map<uint16_t, int64_t> intel_on_enemy_milli;
};

struct WorldSnapshot {
    uint64_t tick = 0;
    std::vector<CountrySnapshot> countries;
};

struct ConcurrentAction {
    Strategy strategy = Strategy::Defend;
    uint16_t target_country_id = 0;
    NegotiationTerms terms;
    float commitment = 0.0f;
};

struct ModelDecision {
    Strategy strategy = Strategy::Defend;
    uint16_t actor_country_id = 0;
    uint16_t target_country_id = 0;
    NegotiationTerms terms;
    std::array<float, 3> allocation = {0.34f, 0.33f, 0.33f};
    float force_commitment = 0.5f;
    bool has_secondary_action = false;
    ConcurrentAction secondary_action;
};

class Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
    virtual Tensor backward(const Tensor& grad_output, const Tensor& pre_activation) const = 0;
};

ModelConfig load_model_config(const std::string& config_path);
bool validate_model_config(const ModelConfig& config, std::string& error_message);
bool inspect_model_state(const std::string& path,
                         size_t* input_dim,
                         size_t* output_dim,
                         ModelConfig* model_config,
                         std::string* error_message);

class Model {
public:
    explicit Model(size_t input_dim, size_t output_dim, const ModelConfig& config);

    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output);
    void configure_optimizer(const OptimizerConfig& config);
    void update(float lr);
    void zero_grad();
    void set_training(bool is_training);
    void set_inference_only(bool enabled);
    bool is_inference_only() const;
    void save(const std::string& path, size_t input_dim, size_t output_dim) const;
    void save_state(const std::string& path,
                    size_t input_dim,
                    size_t output_dim,
                    const ModelConfig& model_config,
                    const ModelTrainingMetadata& metadata) const;
    bool load_state(const std::string& path,
                    size_t input_dim,
                    size_t output_dim,
                    ModelFileInfo* info = nullptr);
    ModelDecision decide(const WorldSnapshot& world_snapshot, uint16_t controlled_country_id);
    size_t input_dim() const;
    size_t output_dim() const;
    void copy_parameters_from(const Model& other);
    size_t gradient_size() const;
    void gradients_to_vector(std::vector<float>& out) const;
    void set_gradients_from_vector(const std::vector<float>& values);

private:
    Tensor apply_norm_forward(const Tensor& x, size_t layer_idx);
    Tensor apply_norm_backward(const Tensor& grad_output, size_t layer_idx);

    std::vector<Linear> layers_;
    std::vector<std::unique_ptr<Activation>> activations_;

    std::vector<Tensor> gamma_;
    std::vector<Tensor> beta_;

    std::vector<Tensor> layer_inputs_;
    std::vector<Tensor> norm_outputs_;
    std::vector<Tensor> dropout_masks_;
    std::vector<Tensor> norm_xhat_cache_;
    std::vector<Tensor> norm_var_cache_;

    std::string norm_type_;
    bool use_norm_;
    bool use_dropout_;
    float dropout_prob_;
    bool training_;
    bool inference_only_;
    float eps_;

    OptimizerConfig optimizer_config_;
    std::vector<float> adam_m_;
    std::vector<float> adam_v_;
    uint64_t adam_t_;
    std::mt19937 dropout_rng_;
    std::uniform_real_distribution<float> dropout_dist_;
};

#endif