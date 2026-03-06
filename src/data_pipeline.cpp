#include "data_pipeline.h"
#include "strategy_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

bool file_exists(const std::string& path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

const char* action_to_name(uint32_t action) {
    switch (action) {
        case battle_common::kActionAttack: return "attack";
        case battle_common::kActionDefend: return "defend";
        case battle_common::kActionNegotiate: return "negotiate";
        case battle_common::kActionSurrender: return "surrender";
        case battle_common::kActionTransferWeapons: return "transfer_weapons";
        case battle_common::kActionFocusEconomy: return "focus_economy";
        case battle_common::kActionDevelopTechnology: return "develop_technology";
        case battle_common::kActionFormAlliance: return "form_alliance";
        case battle_common::kActionBetray: return "betray";
        case battle_common::kActionCyberOperation: return "cyber_operation";
        case battle_common::kActionSignTradeAgreement: return "sign_trade_agreement";
        case battle_common::kActionCancelTradeAgreement: return "cancel_trade_agreement";
        case battle_common::kActionImposeEmbargo: return "impose_embargo";
        case battle_common::kActionInvestInResourceExtraction: return "invest_in_resource_extraction";
        case battle_common::kActionReduceMilitaryUpkeep: return "reduce_military_upkeep";
        case battle_common::kActionSuppressDissent: return "suppress_dissent";
        case battle_common::kActionHoldElections: return "hold_elections";
        case battle_common::kActionCoupAttempt: return "coup_attempt";
        case battle_common::kActionProposeDefensePact: return "propose_defense_pact";
        case battle_common::kActionProposeNonAggression: return "propose_non_aggression";
        case battle_common::kActionBreakTreaty: return "break_treaty";
        case battle_common::kActionRequestIntel: return "request_intel";
        case battle_common::kActionDeployUnits: return "deploy_units";
        case battle_common::kActionTacticalNuke: return "tactical_nuke";
        case battle_common::kActionStrategicNuke: return "strategic_nuke";
        case battle_common::kActionCyberAttack: return "cyber_attack";
        default: return "defend";
    }
}

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float sample_noise(std::mt19937& rng, float sigma) {
    std::normal_distribution<float> dist(0.0f, sigma);
    return dist(rng);
}

uint32_t sample_action_from_scores(std::mt19937& rng,
                                   const std::array<float, battle_common::kBattleOutputDim>& scores) {
    std::array<float, battle_common::kBattleOutputDim> probs{};
    float max_score = scores[0];
    for (size_t i = 1; i < scores.size(); ++i) {
        max_score = std::max(max_score, scores[i]);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        probs[i] = std::exp(scores[i] - max_score);
        sum += probs[i];
    }
    if (sum <= 0.0f) {
        return battle_common::kActionDefend;
    }

    for (float& p : probs) {
        p /= sum;
    }

    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    const float r = unit(rng);
    float cdf = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cdf += probs[i];
        if (r <= cdf || i + 1 == probs.size()) {
            return static_cast<uint32_t>(i);
        }
    }
    return battle_common::kActionDefend;
}

uint32_t action_from_json(const nlohmann::json& value) {
    if (value.is_number_integer()) {
        const int v = value.get<int>();
        if (v >= 0 && v < static_cast<int>(battle_common::kBattleOutputDim)) {
            return static_cast<uint32_t>(v);
        }
        return battle_common::kActionDefend;
    }

    if (!value.is_string()) {
        return battle_common::kActionDefend;
    }

    return pallas::strategy::action_from_string(value.get<std::string>());
}

struct ClusterProfile {
    std::array<float, battle_common::kBattleInputDim> base{};
    std::array<float, 6> latent_mean{};
    std::array<float, 6> latent_sigma{};
};

enum class RareEvent : uint32_t {
    None = 0,
    SurpriseAttack = 1,
    EconomicCollapse = 2,
    CyberBlackout = 3,
    HarshWinter = 4,
};

std::array<ClusterProfile, 10> build_cluster_profiles() {
    std::array<ClusterProfile, 10> c{};

    for (size_t i = 0; i < c.size(); ++i) {
        ClusterProfile p;
        const float bias = static_cast<float>(i) / static_cast<float>(c.size() - 1);

        p.base[0] = 0.35f + 0.35f * (1.0f - bias);
        p.base[1] = 0.30f + 0.30f * bias;
        p.base[2] = clamp01(p.base[0] - p.base[1] + 0.5f);
        p.base[3] = 0.45f + 0.25f * (1.0f - std::abs(0.5f - bias));
        p.base[4] = 0.42f + 0.32f * (1.0f - bias * 0.8f);
        p.base[5] = 0.40f + 0.35f * bias;
        p.base[6] = 0.45f + 0.20f * (1.0f - bias);
        p.base[7] = 0.28f + 0.25f * bias;
        p.base[8] = 0.50f + 0.24f * (1.0f - bias);
        p.base[9] = 0.48f + 0.20f * bias;
        p.base[10] = 0.52f + 0.20f * (1.0f - bias);
        p.base[11] = 0.45f + 0.22f * bias;
        p.base[12] = 0.49f + 0.18f * (1.0f - bias);

        p.base[13] = 0.30f + 0.35f * bias;
        p.base[14] = 0.45f + 0.30f * std::sin(bias * 3.1415926f) * 0.5f;
        p.base[15] = 0.44f + 0.24f * (1.0f - bias);

        p.base[16] = 0.20f + 0.30f * bias;
        p.base[17] = 0.25f + 0.25f * (1.0f - std::abs(0.5f - bias));
        p.base[18] = 0.18f + 0.42f * (1.0f - bias);

        p.base[19] = 0.40f + 0.35f * (1.0f - bias);
        p.base[20] = 0.38f + 0.35f * bias;
        p.base[21] = 0.36f + 0.33f * (1.0f - std::abs(0.5f - bias));
        p.base[22] = 0.42f + 0.30f * (1.0f - bias * 0.6f);

        p.base[23] = 0.25f + 0.45f * (1.0f - bias);
        p.base[24] = 0.30f + 0.40f * bias;
        p.base[25] = 0.40f + 0.30f * (1.0f - std::abs(0.5f - bias));
        p.base[26] = 0.18f + 0.28f * bias;

        p.base[27] = 0.48f + 0.25f * (1.0f - bias);
        p.base[28] = 0.25f + 0.40f * bias;
        p.base[29] = 0.28f + 0.32f * bias;

        p.base[30] = 0.30f + 0.40f * (1.0f - bias);
        p.base[31] = 0.40f + 0.30f * (1.0f - std::abs(0.5f - bias));
        p.base[32] = 0.38f + 0.30f * (1.0f - bias);
        p.base[33] = 0.22f + 0.40f * bias;
        p.base[34] = 0.28f + 0.34f * (1.0f - std::abs(0.5f - bias));
        p.base[35] = 0.18f + 0.42f * (1.0f - bias);
        p.base[36] = 0.20f + 0.35f * (1.0f - bias * 0.8f);
        p.base[37] = 0.42f + 0.35f * bias;
        p.base[38] = 0.25f + 0.38f * (1.0f - bias);
        p.base[39] = 0.35f + 0.30f * (1.0f - std::abs(0.5f - bias));

        p.base[40] = 0.45f + 0.30f * (1.0f - bias);
        p.base[41] = 0.20f + 0.45f * (1.0f - std::abs(0.5f - bias));
        p.base[42] = 0.42f + 0.38f * (1.0f - bias);
        p.base[43] = 0.45f + 0.32f * bias;
        p.base[44] = 0.48f + 0.26f * (1.0f - std::abs(0.5f - bias));
        p.base[45] = 0.22f + 0.36f * bias;
        p.base[46] = 0.22f + 0.38f * bias;
        p.base[47] = 0.28f + 0.42f * bias;
        p.base[48] = 0.36f + 0.32f * (1.0f - std::abs(0.5f - bias));
        p.base[49] = 0.40f + 0.34f * (1.0f - bias);
        p.base[50] = 0.12f + 0.40f * bias;
        p.base[51] = 0.18f + 0.46f * bias;
        p.base[52] = 0.25f + 0.38f * bias;
        p.base[53] = 0.16f + 0.42f * bias;
        p.base[54] = 0.20f + 0.40f * bias;

        p.latent_mean = {
            0.55f * (p.base[0] - p.base[1]),
            0.55f * (p.base[4] + p.base[10] - 1.0f),
            0.50f * (p.base[20] + p.base[27] - 1.0f),
            0.55f * (p.base[23] + p.base[24] - 1.0f),
            0.50f * (p.base[40] + p.base[48] - 1.0f),
            0.55f * (p.base[47] + p.base[50] - 1.0f)
        };
        p.latent_sigma = {0.18f, 0.16f, 0.17f, 0.15f, 0.14f, 0.14f};

        c[i] = p;
    }

    return c;
}

std::array<float, 6> sample_latent(std::mt19937& rng,
                                   const std::array<float, 6>& mean,
                                   const std::array<float, 6>& sigma) {
    std::array<float, 6> out{};
    for (size_t i = 0; i < out.size(); ++i) {
        std::normal_distribution<float> dist(mean[i], sigma[i]);
        out[i] = std::clamp(dist(rng), -2.5f, 2.5f);
    }
    return out;
}

std::array<float, battle_common::kBattleInputDim> sample_state(std::mt19937& rng,
                                                                const ClusterProfile& cluster,
                                                                const std::array<float, 6>& latent,
                                                                float progress,
                                                                float escalation,
                                                                RareEvent event) {
    std::array<float, battle_common::kBattleInputDim> f = cluster.base;

    const float sec = latent[0];
    const float eco = latent[1];
    const float inst = latent[2];
    const float res = latent[3];
    const float trade = latent[4];
    const float polity = latent[5];

    f[0] = clamp01(cluster.base[0] + 0.20f * sec + 0.05f * escalation + sample_noise(rng, 0.03f));
    f[1] = clamp01(cluster.base[1] - 0.12f * sec + 0.08f * escalation + sample_noise(rng, 0.03f));
    f[2] = clamp01(0.5f + 0.4f * (f[0] - f[1]) + 0.12f * (f[8] - f[7]) + sample_noise(rng, 0.03f));
    f[3] = clamp01(cluster.base[3] + 0.18f * eco - 0.10f * escalation + sample_noise(rng, 0.03f));
    f[4] = clamp01(cluster.base[4] + 0.22f * eco - 0.07f * progress + sample_noise(rng, 0.03f));
    f[5] = clamp01(cluster.base[5] + 0.18f * inst + 0.05f * progress + sample_noise(rng, 0.03f));
    f[6] = clamp01(cluster.base[6] + 0.14f * inst + sample_noise(rng, 0.03f));
    f[7] = clamp01(cluster.base[7] + 0.20f * escalation + 0.10f * progress + sample_noise(rng, 0.03f));
    f[8] = clamp01(cluster.base[8] + 0.16f * sec + 0.08f * inst + sample_noise(rng, 0.03f));
    f[9] = clamp01(cluster.base[9] + 0.15f * eco - 0.04f * escalation + sample_noise(rng, 0.03f));
    f[10] = clamp01(cluster.base[10] + 0.12f * inst + 0.10f * eco + sample_noise(rng, 0.03f));
    f[11] = clamp01(cluster.base[11] + 0.11f * eco + 0.06f * inst + sample_noise(rng, 0.03f));
    f[12] = clamp01(cluster.base[12] + 0.14f * res + 0.06f * eco + sample_noise(rng, 0.03f));

    f[13] = clamp01(cluster.base[13] + 0.10f * escalation + 0.06f * progress + sample_noise(rng, 0.03f));
    f[14] = clamp01(cluster.base[14] + 0.09f * std::sin(progress * 6.2831853f) + sample_noise(rng, 0.02f));
    f[15] = clamp01(cluster.base[15] + 0.18f * res - 0.10f * escalation + sample_noise(rng, 0.03f));

    f[16] = clamp01(cluster.base[16] + sample_noise(rng, 0.03f));
    f[17] = clamp01(cluster.base[17] + sample_noise(rng, 0.03f));
    f[18] = clamp01(cluster.base[18] + sample_noise(rng, 0.03f));

    f[19] = clamp01(cluster.base[19] + 0.16f * eco + 0.10f * inst + sample_noise(rng, 0.03f));
    f[20] = clamp01(cluster.base[20] + 0.18f * inst + 0.10f * sec + sample_noise(rng, 0.03f));
    f[21] = clamp01(cluster.base[21] + 0.16f * inst + sample_noise(rng, 0.03f));
    f[22] = clamp01(cluster.base[22] + 0.14f * eco + 0.08f * sec + sample_noise(rng, 0.03f));

    f[23] = clamp01(cluster.base[23] + 0.20f * res - 0.05f * escalation + sample_noise(rng, 0.03f));
    f[24] = clamp01(cluster.base[24] + 0.18f * res + sample_noise(rng, 0.03f));
    f[25] = clamp01(cluster.base[25] + 0.14f * res + 0.05f * inst + sample_noise(rng, 0.03f));
    f[26] = clamp01(cluster.base[26] + 0.12f * res + 0.10f * eco + sample_noise(rng, 0.03f));

    f[27] = clamp01(cluster.base[27] + 0.16f * inst + 0.08f * eco - 0.07f * escalation + sample_noise(rng, 0.03f));
    f[28] = clamp01(cluster.base[28] + 0.20f * escalation - 0.08f * inst + sample_noise(rng, 0.03f));
    f[29] = clamp01(cluster.base[29] + 0.16f * escalation - 0.06f * eco + sample_noise(rng, 0.03f));

    f[30] = clamp01(cluster.base[30] + 0.20f * sec + 0.10f * escalation + sample_noise(rng, 0.03f));
    f[31] = clamp01(cluster.base[31] + 0.10f * sec + 0.08f * inst + sample_noise(rng, 0.03f));
    f[32] = clamp01(cluster.base[32] + 0.20f * (1.0f - escalation) + sample_noise(rng, 0.03f));
    f[33] = clamp01(cluster.base[33] + 0.20f * escalation - 0.08f * inst + sample_noise(rng, 0.03f));
    f[34] = clamp01(cluster.base[34] + 0.18f * (1.0f - escalation) + sample_noise(rng, 0.03f));
    f[35] = clamp01(cluster.base[35] + 0.16f * sec + 0.10f * eco + sample_noise(rng, 0.03f));
    f[36] = clamp01(cluster.base[36] + 0.14f * sec + sample_noise(rng, 0.03f));
    f[37] = clamp01(cluster.base[37] + 0.15f * escalation + 0.10f * progress + sample_noise(rng, 0.03f));
    f[38] = clamp01(cluster.base[38] + 0.20f * sec + 0.08f * res - 0.08f * escalation + sample_noise(rng, 0.03f));
    f[39] = clamp01(cluster.base[39] + 0.09f * std::cos(progress * 6.2831853f) + sample_noise(rng, 0.03f));

    f[40] = clamp01(cluster.base[40] + 0.22f * trade + 0.08f * eco - 0.12f * escalation + sample_noise(rng, 0.03f));
    f[41] = clamp01(cluster.base[41] + 0.26f * trade + 0.06f * polity + sample_noise(rng, 0.03f));
    f[42] = clamp01(cluster.base[42] + 0.20f * res - 0.10f * escalation + sample_noise(rng, 0.03f));
    f[43] = clamp01(cluster.base[43] + 0.18f * res + 0.06f * eco + sample_noise(rng, 0.03f));
    f[44] = clamp01(cluster.base[44] + 0.18f * res - 0.08f * escalation + sample_noise(rng, 0.03f));
    f[45] = clamp01(cluster.base[45] + 0.22f * res + 0.08f * inst + sample_noise(rng, 0.03f));
    f[46] = clamp01(cluster.base[46] + 0.24f * sec + 0.12f * escalation + sample_noise(rng, 0.03f));
    f[47] = clamp01(cluster.base[47] + 0.24f * sec + 0.12f * escalation - 0.10f * polity + sample_noise(rng, 0.03f));
    f[48] = clamp01(cluster.base[48] + 0.22f * eco + 0.16f * trade + sample_noise(rng, 0.03f));
    f[49] = clamp01(cluster.base[49] + 0.18f * polity + 0.14f * eco - 0.10f * escalation + sample_noise(rng, 0.03f));
    f[50] = clamp01(cluster.base[50] + 0.20f * escalation + 0.14f * sec - 0.12f * polity + sample_noise(rng, 0.03f));
    f[51] = clamp01(cluster.base[51] + 0.38f * progress + 0.06f * escalation + sample_noise(rng, 0.03f));
    f[52] = clamp01(cluster.base[52] + 0.26f * escalation + 0.16f * sec - 0.10f * polity + sample_noise(rng, 0.03f));
    f[53] = clamp01(cluster.base[53] + 0.34f * progress + 0.14f * escalation - 0.06f * eco + sample_noise(rng, 0.03f));
    f[54] = clamp01(cluster.base[54] + 0.22f * escalation + 0.12f * sec + 0.10f * (1.0f - res) + sample_noise(rng, 0.03f));

    if (event == RareEvent::SurpriseAttack && progress < 0.35f) {
        f[1] = clamp01(f[1] + 0.25f + sample_noise(rng, 0.02f));
        f[8] = clamp01(f[8] - 0.18f + sample_noise(rng, 0.02f));
        f[33] = clamp01(f[33] + 0.22f + sample_noise(rng, 0.02f));
        f[46] = clamp01(f[46] + 0.16f + sample_noise(rng, 0.02f));
        f[53] = clamp01(f[53] + 0.10f + sample_noise(rng, 0.02f));
    } else if (event == RareEvent::EconomicCollapse && progress > 0.30f) {
        f[3] = clamp01(f[3] - 0.30f + sample_noise(rng, 0.02f));
        f[4] = clamp01(f[4] - 0.28f + sample_noise(rng, 0.02f));
        f[27] = clamp01(f[27] - 0.24f + sample_noise(rng, 0.02f));
        f[28] = clamp01(f[28] + 0.22f + sample_noise(rng, 0.02f));
        f[40] = clamp01(f[40] - 0.26f + sample_noise(rng, 0.02f));
        f[50] = clamp01(f[50] + 0.18f + sample_noise(rng, 0.02f));
    } else if (event == RareEvent::CyberBlackout) {
        f[9] = clamp01(f[9] - 0.26f + sample_noise(rng, 0.02f));
        f[20] = clamp01(f[20] - 0.22f + sample_noise(rng, 0.02f));
        f[7] = clamp01(f[7] + 0.14f + sample_noise(rng, 0.02f));
        f[38] = clamp01(f[38] - 0.16f + sample_noise(rng, 0.02f));
        f[41] = clamp01(f[41] - 0.18f + sample_noise(rng, 0.02f));
    } else if (event == RareEvent::HarshWinter) {
        f[13] = clamp01(f[13] + 0.24f + sample_noise(rng, 0.02f));
        f[15] = clamp01(f[15] - 0.20f + sample_noise(rng, 0.02f));
        f[25] = clamp01(f[25] - 0.16f + sample_noise(rng, 0.02f));
        f[44] = clamp01(f[44] - 0.20f + sample_noise(rng, 0.02f));
        f[54] = clamp01(f[54] + 0.16f + sample_noise(rng, 0.02f));
    }

    f[55] = clamp01(0.24f + 0.54f * sec + 0.08f * escalation + sample_noise(rng, 0.03f));
    f[56] = clamp01(0.14f + 0.36f * (1.0f - sec) + 0.10f * progress + sample_noise(rng, 0.03f));
    f[57] = clamp01(0.5f * (f[55] + f[56]) + sample_noise(rng, 0.02f));
    f[58] = clamp01(0.5f + 0.45f * (f[0] - f[55]) + sample_noise(rng, 0.02f));
    f[59] = clamp01(0.22f + 0.48f * inst + 0.10f * trade + sample_noise(rng, 0.03f));
    f[60] = clamp01(0.06f + 0.60f * (1.0f - inst) + 0.10f * escalation + sample_noise(rng, 0.03f));
    f[61] = clamp01(0.5f * (f[59] + (1.0f - f[60])) + sample_noise(rng, 0.02f));
    f[62] = clamp01(0.28f + 0.34f * trade + 0.20f * f[49] + sample_noise(rng, 0.03f));
    f[63] = clamp01(0.26f + 0.32f * polity + 0.18f * f[49] + 0.10f * (1.0f - escalation) + sample_noise(rng, 0.03f));
    f[64] = clamp01(0.24f + 0.32f * sec + 0.22f * escalation + sample_noise(rng, 0.03f));
    f[65] = clamp01(0.35f * f[58] + 0.20f * f[59] + 0.15f * f[8] + 0.10f * f[86] + 0.12f * f[89] - 0.08f * f[51] + sample_noise(rng, 0.03f));
    f[66] = clamp01(0.26f * (1.0f - f[10]) + 0.24f * (1.0f - f[9]) + 0.20f * f[45] + 0.16f * f[47] + 0.14f * f[88] + sample_noise(rng, 0.03f));
    f[67] = clamp01(0.18f + 0.28f * sec + 0.18f * inst + 0.10f * escalation + sample_noise(rng, 0.03f));
    f[68] = clamp01(0.20f + 0.26f * sec + 0.16f * polity + 0.12f * f[50] + sample_noise(rng, 0.03f));
    f[69] = clamp01(0.26f + 0.42f * polity + 0.14f * trade - 0.12f * escalation + sample_noise(rng, 0.03f));
    f[70] = clamp01(progress + sample_noise(rng, 0.02f));
    f[71] = clamp01(0.65f + 0.20f * sec + sample_noise(rng, 0.02f));
    f[72] = clamp01(0.12f + 0.20f * sec + 0.18f * escalation + sample_noise(rng, 0.03f));
    f[73] = clamp01(0.18f + 0.26f * escalation + 0.12f * sec + sample_noise(rng, 0.03f));
    f[74] = clamp01(0.20f + 0.24f * (1.0f - f[7]) + 0.14f * sec + sample_noise(rng, 0.03f));
    f[75] = clamp01(0.12f + 0.20f * trade + 0.18f * polity + sample_noise(rng, 0.03f));
    f[76] = clamp01(0.14f + 0.24f * sec + 0.20f * res + sample_noise(rng, 0.03f));
    f[77] = clamp01(0.16f + 0.26f * sec + 0.22f * inst + sample_noise(rng, 0.03f));
    f[78] = clamp01(0.18f + 0.22f * f[46] + 0.16f * (1.0f - 0.25f * (f[34] + f[35] + f[36] + f[37])) + 0.10f * f[52] + sample_noise(rng, 0.03f));
    f[79] = clamp01(0.14f + 0.28f * sec + 0.20f * (1.0f - f[7]) + sample_noise(rng, 0.03f));
    f[80] = clamp01(f[55] * 0.90f + 0.10f * (1.0f - f[86]) + sample_noise(rng, 0.02f));
    f[81] = clamp01(f[56] * 0.88f + 0.08f * f[85] + sample_noise(rng, 0.02f));
    f[82] = clamp01(0.5f * (f[80] + f[81]) + sample_noise(rng, 0.02f));
    f[83] = clamp01(0.08f + 0.46f * (1.0f - polity) + 0.14f * escalation + sample_noise(rng, 0.03f));
    f[84] = clamp01(0.24f + 0.42f * polity + 0.10f * trade + sample_noise(rng, 0.03f));
    f[85] = clamp01(0.5f * (f[83] + f[84]) + sample_noise(rng, 0.02f));
    f[86] = clamp01(0.20f + 0.42f * inst + 0.16f * trade + sample_noise(rng, 0.03f));
    f[87] = clamp01(0.12f + 0.50f * inst + 0.14f * sec + sample_noise(rng, 0.03f));
    f[88] = clamp01(0.10f + 0.28f * escalation + 0.24f * sec + sample_noise(rng, 0.03f));
    f[89] = clamp01(0.18f + 0.26f * f[86] + 0.18f * f[88] + 0.16f * f[58] + 0.08f * trade + sample_noise(rng, 0.03f));

    return f;
}

std::array<float, battle_common::kBattleOutputDim> score_actions(
    std::mt19937& rng,
    const std::array<float, battle_common::kBattleInputDim>& f,
    float progress) {
    std::array<float, battle_common::kBattleOutputDim> s{};

    s[battle_common::kActionAttack] =
        1.9f * f[0] + 1.1f * f[2] + 0.9f * f[8] + 0.7f * f[15] + 0.6f * f[31] -
        0.8f * f[47] + 0.5f * f[52] - 1.0f * f[7] - 0.7f * f[53] + sample_noise(rng, 0.13f);

    s[battle_common::kActionDefend] =
        1.7f * f[1] + 0.8f * f[4] + 0.7f * f[7] + 0.7f * f[15] + 0.6f * f[19] -
        0.5f * f[33] + sample_noise(rng, 0.13f);

    s[battle_common::kActionNegotiate] =
        1.3f * f[5] + 0.9f * f[32] + 0.8f * f[27] - 0.9f * f[33] - 0.6f * f[37] +
        sample_noise(rng, 0.13f);

    s[battle_common::kActionSurrender] =
        1.8f * (1.0f - f[3]) + 1.0f * (1.0f - f[8]) + 1.0f * f[7] + 0.7f * f[28] -
        0.9f * f[0] + 0.6f * f[50] + 0.4f * progress + sample_noise(rng, 0.14f);

    s[battle_common::kActionTransferWeapons] =
        1.2f * f[12] + 1.0f * f[23] + 0.7f * f[24] + 0.6f * f[35] - 0.8f * f[8] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionFocusEconomy] =
        1.6f * f[3] + 1.2f * f[4] + 0.8f * f[10] + 0.7f * f[27] + 0.5f * f[25] +
        0.6f * f[48] - 0.7f * f[33] + sample_noise(rng, 0.12f);

    s[battle_common::kActionDevelopTechnology] =
        1.4f * f[11] + 1.1f * f[19] + 1.2f * f[20] + 0.9f * f[22] + 0.6f * f[26] -
        0.6f * f[28] + sample_noise(rng, 0.12f);

    s[battle_common::kActionFormAlliance] =
        1.6f * f[32] + 1.1f * f[5] + 0.8f * f[27] + 0.7f * f[34] - 1.0f * f[33] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionBetray] =
        1.4f * f[33] + 1.1f * f[0] + 0.8f * f[20] + 0.7f * f[31] - 1.0f * f[32] +
        sample_noise(rng, 0.13f);

    s[battle_common::kActionCyberOperation] =
        1.7f * f[20] + 1.1f * f[21] + 0.8f * f[9] + 0.7f * f[38] - 0.7f * f[15] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionSignTradeAgreement] =
        1.6f * f[40] + 1.3f * f[41] + 0.9f * f[48] + 0.8f * f[49] - 0.8f * f[47] -
        0.6f * f[50] + sample_noise(rng, 0.12f);

    s[battle_common::kActionCancelTradeAgreement] =
        1.2f * (1.0f - f[40]) + 0.9f * f[41] + 0.8f * f[46] + 0.7f * f[54] -
        0.7f * f[49] + sample_noise(rng, 0.12f);

    s[battle_common::kActionImposeEmbargo] =
        1.2f * f[47] + 1.0f * f[40] + 0.8f * f[20] + 0.8f * f[46] - 0.8f * f[49] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionInvestInResourceExtraction] =
        1.4f * (1.0f - f[42]) + 1.3f * (1.0f - f[43]) + 1.0f * (1.0f - f[44]) + 0.9f * (1.0f - f[45]) +
        0.8f * f[48] - 0.7f * f[46] + sample_noise(rng, 0.12f);

    s[battle_common::kActionReduceMilitaryUpkeep] =
        1.4f * f[46] + 1.0f * f[53] + 0.9f * f[50] + 0.8f * (1.0f - f[0]) - 0.9f * f[47] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionSuppressDissent] =
        1.5f * f[28] + 1.1f * f[50] + 0.8f * f[47] - 0.7f * f[49] + sample_noise(rng, 0.12f);

    s[battle_common::kActionHoldElections] =
        1.6f * f[51] + 1.2f * f[49] + 0.8f * f[27] - 0.7f * f[47] - 0.6f * f[50] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionCoupAttempt] =
        1.8f * f[50] + 1.2f * f[47] + 0.9f * (1.0f - f[27]) + 0.8f * (1.0f - f[49]) -
        1.0f * f[41] + sample_noise(rng, 0.14f);

    s[battle_common::kActionProposeDefensePact] =
        1.6f * f[63] + 1.1f * f[84] + 0.8f * f[85] + 0.7f * f[50] - 0.7f * f[66] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionProposeNonAggression] =
        1.5f * f[63] + 1.0f * f[84] + 0.8f * f[85] + 0.6f * f[51] - 0.6f * f[65] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionBreakTreaty] =
        1.4f * f[88] + 1.1f * f[65] + 0.9f * f[64] - 0.9f * f[84] - 0.7f * f[85] +
        sample_noise(rng, 0.13f);

    s[battle_common::kActionRequestIntel] =
        1.7f * (1.0f - f[61]) + 1.0f * f[74] + 0.9f * f[86] + 0.7f * f[89] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionDeployUnits] =
        1.5f * f[65] + 1.1f * f[58] + 0.9f * f[68] + 0.8f * f[76] - 0.8f * f[66] +
        sample_noise(rng, 0.12f);

    s[battle_common::kActionTacticalNuke] =
        1.6f * f[67] + 1.2f * f[68] + 1.0f * f[65] + 0.8f * f[73] - 1.2f * f[84] +
        sample_noise(rng, 0.15f);

    s[battle_common::kActionStrategicNuke] =
        1.8f * f[67] + 1.2f * f[68] + 1.1f * f[73] + 0.9f * f[88] - 1.4f * f[84] -
        0.8f * f[85] + sample_noise(rng, 0.16f);

    s[battle_common::kActionCyberAttack] =
        1.6f * f[20] + 1.1f * f[21] + 1.0f * f[65] + 0.8f * f[74] + 0.8f * f[86] -
        0.7f * f[66] + sample_noise(rng, 0.12f);

    return s;
}

bool validate_sample_heuristics(const std::array<float, battle_common::kBattleInputDim>& f,
                                uint32_t action,
                                float tolerance) {
    if (action >= battle_common::kBattleOutputDim) {
        return false;
    }

    for (float v : f) {
        if (v < -0.0001f || v > 1.0001f) {
            return false;
        }
    }

    if (f[16] + f[17] + f[18] > 1.8f) {
        return false;
    }

    const float expected_balance = clamp01(0.5f + 0.45f * (f[0] - f[1]) + 0.10f * (f[8] - f[7]));
    if (std::abs(f[2] - expected_balance) > tolerance) {
        return false;
    }

    const float reserve_mean = 0.25f * (f[42] + f[43] + f[44] + f[45]);
    if (std::abs(f[54] - clamp01(0.22f + 0.18f * f[46] + 0.16f * (1.0f - reserve_mean) + 0.10f * f[52])) > 0.34f) {
        return false;
    }

    if (action == battle_common::kActionSurrender && f[0] > 0.72f && f[8] > 0.66f && f[7] < 0.42f) {
        return false;
    }
    if (action == battle_common::kActionAttack && f[0] < 0.28f && f[8] < 0.30f && f[7] > 0.60f) {
        return false;
    }
    if (action == battle_common::kActionFocusEconomy && (f[3] < 0.24f || f[4] < 0.24f)) {
        return false;
    }
    if (action == battle_common::kActionDevelopTechnology && (f[11] < 0.22f || f[19] < 0.25f)) {
        return false;
    }
    if (action == battle_common::kActionCyberOperation && f[20] < 0.22f) {
        return false;
    }
    if (action == battle_common::kActionSignTradeAgreement && (f[40] < 0.28f || f[41] < 0.18f)) {
        return false;
    }
    if (action == battle_common::kActionInvestInResourceExtraction && reserve_mean > 0.78f) {
        return false;
    }
    if (action == battle_common::kActionReduceMilitaryUpkeep && (f[46] < 0.24f || f[53] < 0.20f)) {
        return false;
    }
    if (action == battle_common::kActionSuppressDissent && f[28] < 0.28f) {
        return false;
    }
    if (action == battle_common::kActionHoldElections && f[51] < 0.24f) {
        return false;
    }
    if (action == battle_common::kActionCoupAttempt && (f[50] < 0.42f || f[47] < 0.34f)) {
        return false;
    }
    if (action == battle_common::kActionProposeDefensePact && (f[63] < 0.26f || f[84] < 0.26f)) {
        return false;
    }
    if (action == battle_common::kActionProposeNonAggression && (f[63] < 0.24f || f[84] < 0.24f)) {
        return false;
    }
    if (action == battle_common::kActionBreakTreaty && (f[88] < 0.20f || f[65] < 0.24f)) {
        return false;
    }
    if (action == battle_common::kActionRequestIntel && (f[61] > 0.86f || f[86] < 0.20f)) {
        return false;
    }
    if (action == battle_common::kActionDeployUnits && (f[65] < 0.24f || f[68] < 0.18f)) {
        return false;
    }
    if (action == battle_common::kActionTacticalNuke && (f[67] < 0.40f || f[84] > 0.82f)) {
        return false;
    }
    if (action == battle_common::kActionStrategicNuke && (f[67] < 0.56f || f[84] > 0.74f)) {
        return false;
    }
    if (action == battle_common::kActionCyberAttack && (f[20] < 0.24f || f[86] < 0.22f)) {
        return false;
    }

    return true;
}

nlohmann::json synthesize_dataset(const BattleDatasetConfig& config) {
    std::mt19937 rng(config.rng_seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    const std::array<ClusterProfile, 10> all_profiles = build_cluster_profiles();

    const size_t cluster_count = std::clamp<size_t>(config.cluster_count, 8, 12);
    const size_t sequence_length = std::max<size_t>(3, config.sequence_length);
    const size_t target_samples = std::max<size_t>(config.synthetic_samples, sequence_length);
    const float edge_case_rate = std::clamp(config.edge_case_rate, 0.0f, 0.30f);
    const float validation_tolerance = std::clamp(config.validation_tolerance, 0.08f, 0.30f);

    std::uniform_int_distribution<size_t> cluster_dist(0, std::min(all_profiles.size(), cluster_count) - 1);
    std::uniform_real_distribution<float> escalation_dist(-0.22f, 0.35f);
    std::uniform_int_distribution<int> rare_event_dist(1, 4);
    std::normal_distribution<float> latent_rw(0.0f, 0.035f);

    nlohmann::json root = nlohmann::json::array();

    size_t sequence_id = 0;
    size_t attempts = 0;
    const size_t max_attempts = target_samples * 120;

    while (root.size() < target_samples && attempts < max_attempts) {
        ++attempts;
        const size_t cluster_id = cluster_dist(rng);
        const ClusterProfile& cluster = all_profiles[cluster_id];

        std::array<float, 6> latent = sample_latent(rng, cluster.latent_mean, cluster.latent_sigma);
        const float escalation = escalation_dist(rng);

        RareEvent event = RareEvent::None;
        if (unit(rng) < edge_case_rate) {
            event = static_cast<RareEvent>(rare_event_dist(rng));
        }

        for (size_t step = 0; step < sequence_length && root.size() < target_samples; ++step) {
            const float progress = sequence_length > 1
                ? static_cast<float>(step) / static_cast<float>(sequence_length - 1)
                : 0.0f;

            for (float& lv : latent) {
                lv = std::clamp(lv + latent_rw(rng), -2.5f, 2.5f);
            }

            std::array<float, battle_common::kBattleInputDim> f = sample_state(
                rng,
                cluster,
                latent,
                progress,
                escalation,
                event);

            const std::array<float, battle_common::kBattleOutputDim> action_scores = score_actions(rng, f, progress);
            uint32_t action = sample_action_from_scores(rng, action_scores);
            if (!validate_sample_heuristics(f, action, validation_tolerance)) {
                continue;
            }

            nlohmann::json item;
            item["features"] = nlohmann::json::array();
            for (float x : f) {
                item["features"].push_back(x);
            }
            item["action"] = action;
            item["sequence_id"] = sequence_id;
            item["step"] = step;
            item["cluster_id"] = cluster_id;
            item["rare_event"] = static_cast<uint32_t>(event);
            root.push_back(std::move(item));
        }
        ++sequence_id;
    }

    if (root.size() < target_samples) {
        std::ostringstream oss;
        oss << "Failed to synthesize requested battle dataset size. requested=" << target_samples
            << " generated=" << root.size();
        throw std::runtime_error(oss.str());
    }

    return root;
}

bool parse_json_samples(const nlohmann::json& root, std::vector<BattleSample>& samples) {
    if (!root.is_array()) {
        return false;
    }

    samples.clear();
    samples.reserve(root.size());
    for (const auto& item : root) {
        if (!item.is_object() || !item.contains("features") || !item["features"].is_array() || !item.contains("action")) {
            continue;
        }
        const auto& features = item["features"];
        if (features.size() != battle_common::kBattleInputDim) {
            continue;
        }

        BattleSample sample;
        bool ok = true;
        for (size_t i = 0; i < battle_common::kBattleInputDim; ++i) {
            if (!features[i].is_number()) {
                ok = false;
                break;
            }
            sample.features[i] = features[i].get<float>();
        }
        if (!ok) {
            continue;
        }

        sample.action = action_from_json(item["action"]);
        if (sample.action >= battle_common::kBattleOutputDim) {
            continue;
        }
        samples.push_back(sample);
    }

    return !samples.empty();
}

void write_txt_dataset(const std::string& txt_path, const std::vector<BattleSample>& samples) {
    std::ofstream out(txt_path);
    if (!out) {
        throw std::runtime_error("Failed to write battle txt dataset: " + txt_path);
    }

    out << "# ";
    for (size_t i = 0; i < battle_common::kBattleInputDim; ++i) {
        if (i > 0) {
            out << ',';
        }
        out << "f" << i;
    }
    out << "|action_id|action_name\n";

    out << std::fixed << std::setprecision(6);
    for (const BattleSample& s : samples) {
        for (size_t i = 0; i < battle_common::kBattleInputDim; ++i) {
            if (i > 0) out << ',';
            out << s.features[i];
        }
        out << '|' << s.action << '|' << action_to_name(s.action) << '\n';
    }
}

void write_vocab(const std::string& vocab_path) {
    std::ofstream out(vocab_path);
    if (!out) {
        throw std::runtime_error("Failed to write vocab file: " + vocab_path);
    }

    for (size_t i = 0; i < battle_common::kBattleInputDim; ++i) {
        out << "f" << i << '\n';
    }

    for (size_t i = 0; i < battle_common::kBattleOutputDim; ++i) {
        out << action_to_name(static_cast<uint32_t>(i)) << '\n';
    }
}

}  // namespace

BattleDatasetInfo prepare_battle_dataset(const BattleDatasetConfig& config,
                                         std::vector<BattleSample>& samples,
                                         bool force_rebuild) {
    nlohmann::json root;
    bool rebuilt = false;

    if (!force_rebuild && file_exists(config.json_path)) {
        std::ifstream in(config.json_path);
        if (in) {
            try {
                in >> root;
            } catch (...) {
                root = nlohmann::json();
            }
        }
    }

    if (force_rebuild || !root.is_array() || root.empty()) {
        root = synthesize_dataset(config);
        std::ofstream out(config.json_path);
        if (!out) {
            throw std::runtime_error("Failed to write battle json dataset: " + config.json_path);
        }
        out << root.dump(2) << '\n';
        rebuilt = true;
    }

    if (!parse_json_samples(root, samples)) {
        throw std::runtime_error("Invalid dataset format: expected array of {features[" + std::to_string(battle_common::kBattleInputDim) + "], action}");
    }

    write_txt_dataset(config.txt_path, samples);
    write_vocab(config.vocab_path);

    BattleDatasetInfo info;
    info.sample_count = samples.size();
    info.rebuilt = rebuilt;
    info.json_path = config.json_path;
    info.txt_path = config.txt_path;
    info.vocab_path = config.vocab_path;
    return info;
}

BattleDatasetConfig make_battle_dataset_config(const std::string& data_root) {
    BattleDatasetConfig config;
    config.data_root = data_root;
    config.json_path = data_root + "/battle_train.json";
    config.txt_path = data_root + "/battle_train.txt";
    config.vocab_path = data_root + "/vocab.txt";
    return config;
}
