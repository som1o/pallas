#ifndef BATTLE_COMMON_H
#define BATTLE_COMMON_H

#include <cstddef>

namespace battle_common {

constexpr size_t kBattleInputDim = 90;
constexpr size_t kBattleOutputDim = 26;

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
