#ifndef BATTLE_COMMON_H
#define BATTLE_COMMON_H

#include <cstddef>

namespace battle_common {

constexpr size_t kBattleBaseInputDim = 90;
constexpr size_t kBattleTemporalWindow = 8;
constexpr size_t kBattleInputDim = kBattleBaseInputDim * kBattleTemporalWindow;

constexpr size_t kBattlePolicyActionDim = 26;
constexpr size_t kBattleStrategicGoalDim = 4;  // attack, defend, trade, develop
constexpr size_t kBattleTacticalTargetBucketDim = 3;
constexpr size_t kBattleCommitmentBucketDim = 3;
constexpr size_t kBattleAllocationBucketDim = 3;
constexpr size_t kBattleOpponentActionDim = 26;
constexpr size_t kBattleValueDim = 1;

constexpr size_t kBattleHeadPolicyOffset = 0;
constexpr size_t kBattleHeadStrategicOffset = kBattleHeadPolicyOffset + kBattlePolicyActionDim;
constexpr size_t kBattleHeadTargetBucketOffset = kBattleHeadStrategicOffset + kBattleStrategicGoalDim;
constexpr size_t kBattleHeadCommitmentOffset = kBattleHeadTargetBucketOffset + kBattleTacticalTargetBucketDim;
constexpr size_t kBattleHeadAllocationOffset = kBattleHeadCommitmentOffset + kBattleCommitmentBucketDim;
constexpr size_t kBattleHeadOpponentOffset = kBattleHeadAllocationOffset + kBattleAllocationBucketDim;
constexpr size_t kBattleHeadValueOffset = kBattleHeadOpponentOffset + kBattleOpponentActionDim;
constexpr size_t kBattleOutputDim = kBattleHeadValueOffset + kBattleValueDim;

enum BattleAction : size_t {
	kActionAttack = 0,
	kActionDefend = 1,
	kActionNegotiate = 2,
	kActionSurrender = 3,
	kActionTransferWeapons = 4,
	kActionFocusEconomy = 5,
	kActionDevelopTechnology = 6,
	kActionFormAlliance = 7,
	kActionBetray = 8,
	kActionCyberOperation = 9,
	kActionSignTradeAgreement = 10,
	kActionCancelTradeAgreement = 11,
	kActionImposeEmbargo = 12,
	kActionInvestInResourceExtraction = 13,
	kActionReduceMilitaryUpkeep = 14,
	kActionSuppressDissent = 15,
	kActionHoldElections = 16,
	kActionCoupAttempt = 17,
	kActionProposeDefensePact = 18,
	kActionProposeNonAggression = 19,
	kActionBreakTreaty = 20,
	kActionRequestIntel = 21,
	kActionDeployUnits = 22,
	kActionTacticalNuke = 23,
	kActionStrategicNuke = 24,
	kActionCyberAttack = 25,
};

}  // namespace battle_common

#endif
