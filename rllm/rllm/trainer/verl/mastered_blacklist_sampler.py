import logging
import random
from typing import Iterable

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class MasteredBlacklistSampler(Sampler[int]):
    def __init__(self, data_source, shuffle: bool = True, seed: int | None = None):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = 0 if seed is None else int(seed)
        self.epoch = 0
        self.mastered_ids: set[str] = set()
        self._index_to_sample_id: list[str] = []
        self._sample_id_to_index: dict[str, int] = {}

        if hasattr(data_source, 'dataframe') and 'id' in data_source.dataframe.column_names:
            for idx, sample_id in enumerate(data_source.dataframe['id']):
                sid = str(sample_id)
                self._index_to_sample_id.append(sid)
                self._sample_id_to_index[sid] = idx
        else:
            has_id = False
            if len(data_source) > 0:
                try:
                    _ = data_source[0]["id"]
                    has_id = True
                except (KeyError, TypeError):
                    pass

            if has_id:
                for idx in range(len(data_source)):
                    sample_id = str(data_source[idx]["id"])
                    self._index_to_sample_id.append(sample_id)
                    self._sample_id_to_index[sample_id] = idx
            else:
                logger.warning(
                    "MasteredBlacklistSampler: no 'id' field found, using index as sample id. "
                    "Add 'id' to your JSONL for stable cross-epoch tracking."
                )
                for idx in range(len(data_source)):
                    sid = str(idx)
                    self._index_to_sample_id.append(sid)
                    self._sample_id_to_index[sid] = idx

        logger.info(
            "MasteredBlacklistSampler: %d samples, first 3 ids: %s",
            len(self._index_to_sample_id), self._index_to_sample_id[:3],
        )
        self._rebuild_active_indices()

    def _rebuild_active_indices(self):
        self.active_indices = [
            idx for idx, sample_id in enumerate(self._index_to_sample_id)
            if sample_id not in self.mastered_ids
        ]

    def mark_mastered(self, sample_ids: Iterable[str]) -> int:
        before = len(self.mastered_ids)
        str_ids = [str(sid) for sid in sample_ids]
        valid_ids = [sid for sid in str_ids if sid in self._sample_id_to_index]
        unknown_ids = [sid for sid in str_ids if sid not in self._sample_id_to_index]
        if valid_ids:
            self.mastered_ids.update(valid_ids)
            self._rebuild_active_indices()
        if unknown_ids:
            logger.warning(
                "MasteredBlacklistSampler.mark_mastered: %d unknown ids ignored (first 3: %s)",
                len(unknown_ids), unknown_ids[:3],
            )
        return len(self.mastered_ids) - before

    def num_active(self) -> int:
        return len(self.active_indices)

    def num_mastered(self) -> int:
        return len(self.mastered_ids)

    def total_size(self) -> int:
        return len(self._index_to_sample_id)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        return {
            "mastered_ids": sorted(self.mastered_ids),
            "epoch": self.epoch,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state: dict):
        self.mastered_ids = set(map(str, state.get("mastered_ids", [])))
        self.epoch = int(state.get("epoch", 0))
        self._rebuild_active_indices()

    def __iter__(self):
        indices = list(self.active_indices)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.active_indices)
