#ifndef DATALOADER_H
#define DATALOADER_H

#include "battle_common.h"
#include "tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

class BattleBatchLoader {
public:
    BattleBatchLoader(const std::vector<std::array<float, battle_common::kBattleInputDim>>* features,
                      const std::vector<uint32_t>* actions,
                      const std::vector<size_t>& indices,
                      size_t batch_size,
                      bool shuffle,
                      uint32_t seed = 1234);

    void reset();
    bool next(Tensor& inputs, std::vector<uint32_t>& targets);
    size_t steps_per_epoch() const;

private:
    const std::vector<std::array<float, battle_common::kBattleInputDim>>* features_;
    const std::vector<uint32_t>* actions_;
    size_t batch_size_;
    std::vector<size_t> indices_;
    size_t cursor_;
    bool shuffle_;
    std::mt19937 rng_;
};

#endif
