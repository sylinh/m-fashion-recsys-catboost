from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class IdMap:
    user_id2index: Dict[str, int]
    user_index2id: Dict[int, str]
    item_id2index: Dict[int, int]
    item_index2id: Dict[int, int]

    @classmethod
    def load(cls, base_dir: Path, map_dir: str = "index_id_map") -> "IdMap":
        d = Path(base_dir) / map_dir
        return cls(
            user_id2index=pickle.load(open(d / "user_id2index.pkl", "rb")),
            user_index2id=pickle.load(open(d / "user_index2id.pkl", "rb")),
            item_id2index=pickle.load(open(d / "item_id2index.pkl", "rb")),
            item_index2id=pickle.load(open(d / "item_index2id.pkl", "rb")),
        )

