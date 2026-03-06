#include "dataloader.h"

#include <algorithm>

BattleBatchLoader::BattleBatchLoader(const std::vector<std::array<float, battle_common::kBattleInputDim>>* features,
                                     const std::vector<uint32_t>* actions,
                                     const std::vector<size_t>& indices,
                                     size_t batch_size,
                                     bool shuffle,
                                     uint32_t seed)
    : features_(features),
      actions_(actions),
      batch_size_(batch_size),
      indices_(indices),
      cursor_(0),
      shuffle_(shuffle),
      rng_(seed) {}

void BattleBatchLoader::reset() {
    cursor_ = 0;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}

bool BattleBatchLoader::next(Tensor& inputs, std::vector<uint32_t>& targets) {
    if (features_ == nullptr || actions_ == nullptr || cursor_ >= indices_.size()) {
        return false;
    }

    const size_t remaining = indices_.size() - cursor_;
    const size_t batch = std::min(batch_size_, remaining);
    inputs = Tensor({batch, battle_common::kBattleInputDim}, 0.0f);
    targets.assign(batch, 0);

    for (size_t row = 0; row < batch; ++row) {
        const size_t sample_idx = indices_[cursor_ + row];
        if (sample_idx >= features_->size() || sample_idx >= actions_->size()) {
            continue;
        }

        const auto& sample = (*features_)[sample_idx];
        for (size_t col = 0; col < battle_common::kBattleInputDim; ++col) {
            inputs.data[row * battle_common::kBattleInputDim + col] = sample[col];
        }
        targets[row] = (*actions_)[sample_idx];
    }

    cursor_ += batch;
    return true;
}

size_t BattleBatchLoader::steps_per_epoch() const {
    if (indices_.empty() || batch_size_ == 0) {
        return 0;
    }
    return (indices_.size() + batch_size_ - 1) / batch_size_;
}
