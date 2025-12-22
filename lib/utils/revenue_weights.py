from dataclasses import dataclass
from typing import Any, Dict
import json
import numpy as np
from collections import defaultdict

@dataclass(frozen=True)
class RevenueWeights:
    weights: Dict[str, Dict[str, float]]
    default_weight: float = 1.0
    normalize: bool = True
    _scale: float = 1.0

    @staticmethod
    def from_json(path: str, default_weight: float = 1.0, normalize: bool = True) -> "RevenueWeights":
        with open(path, "r") as f:
            raw = json.load(f) or {}

        cleaned: Dict[str, Dict[str, float]] = {}
        vals = []
        for prod, seg_map in raw.items():
            if not isinstance(seg_map, dict):
                continue
            prod = str(prod)
            cleaned[prod] = {}
            for seg, w in seg_map.items():
                try:
                    w = float(w)
                except (TypeError, ValueError):
                    continue
                seg = str(seg)
                cleaned[prod][seg] = w
                vals.append(w)

        scale = float(np.mean(vals)) if (normalize and len(vals) > 0) else 1.0
        scale = scale if scale > 0 else 1.0

        return RevenueWeights(weights=cleaned, default_weight=float(default_weight), normalize=normalize, _scale=scale)

    def get(self, product: Any, segment: Any) -> float:
        prod, seg = str(product), str(segment)
        w = float(self.weights.get(prod, {}).get(seg, self.default_weight))
        if self.normalize:
            w = w / self._scale
        return max(w, 0.0)
    

def calculate_revenue_weights(df, out_path:str = "lib/configs/revenue_weights.json" ):
    grp_rev = (
        df.groupby(["dept_id", "customer_segment"], observed=True)["revenue"]
        .sum()
        .sort_index()
    )
    #  Weights
    total_rev = float(grp_rev.sum())
    weights = (grp_rev / total_rev) if total_rev != 0 else grp_rev * 0.0

    #  Build nested dict: dept_id -> customer_segment -> weight
    nested = defaultdict(dict)
    for (dept_id, customer_segment), w in weights.items():
        nested[str(dept_id)][str(customer_segment)] = float(w)

    # Save to JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")