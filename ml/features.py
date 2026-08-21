"""Feature vectors, built the same way at train time and predict time.

One function, both directions. train.py fits on vectors from
backfill.TrainingExample.to_row(); predict.py scores vectors from
candidate_generator.Candidate.to_dict(). Those two dicts share their feature
keys deliberately, and this module is the only thing that turns either into
numbers.

The FeatureSpec is the contract, and it ships inside model_metadata.json next to
the coefficients. It is passed in, never inferred at predict time: a spec derived
from whatever data happens to be in front of you produces a different column
order -- or a different column count -- for the same row, and the coefficients
then land on columns they were never fitted on. There is no error when that
happens. The predictions just quietly stop meaning anything, which is why
`vectorize` refuses to guess about anything it was not given.

Feature budget is small on purpose. With roughly 30 attended days, each one a
choice set of "here were the options, I took this one", you can support 3-4
parameters before the fit is memorizing days rather than learning times. DEFAULT
is deliberately three. Widen it as the season fills in -- add "workout" to
categorical, or "gap_before_minutes" to numeric -- and retrain; nothing else has
to change, because the spec travels with the model.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

WEEKEND = {"Saturday", "Sunday"}


def _is_weekend(row: dict) -> float:
    return 1.0 if row.get("weekday") in WEEKEND else 0.0


# Features computed from a row rather than read off it. Keep them pure and
# total: a derived feature that raises on an unexpected row turns a scoring run
# into an outage.
DERIVED: dict[str, Callable[[dict], float]] = {
    "is_weekend": _is_weekend,
}


@dataclass(frozen=True)
class FeatureSpec:
    """Ordered, frozen description of the vector `vectorize` produces."""

    numeric: tuple[str, ...] = ()
    derived: tuple[str, ...] = ()
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.derived) - set(DERIVED)
        if unknown:
            raise ValueError(f"unknown derived features: {sorted(unknown)}")

    @property
    def feature_names(self) -> list[str]:
        """Column names, in the exact order vectorize emits them."""
        names = list(self.numeric) + list(self.derived)
        for column in sorted(self.categorical):
            names.extend(
                f"{column}={value}" for value in self.categorical[column]
            )
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric": list(self.numeric),
            "derived": list(self.derived),
            "categorical": {
                column: list(values) for column, values in self.categorical.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSpec":
        return cls(
            numeric=tuple(payload.get("numeric", ())),
            derived=tuple(payload.get("derived", ())),
            categorical={
                column: tuple(values)
                for column, values in (payload.get("categorical") or {}).items()
            },
        )


# Three parameters, chosen to fit the data you actually have:
#   start_hour          the thing the model exists to learn
#   gap_after_minutes   how much slack follows the session -- a slot with 20
#                       minutes after it is a slot you skip
#   is_weekend          one column instead of seven for weekday
DEFAULT = FeatureSpec(
    numeric=("start_hour", "gap_after_minutes"),
    derived=("is_weekend",),
)


def vectorize(row: dict, spec: FeatureSpec) -> list[float]:
    """One row to one vector, in spec order.

    Raises KeyError for a missing numeric feature -- that is a schema mismatch
    between the row producer and the spec, and silently imputing zero would hide
    it. An unrecognized *category* is different and not an error: it encodes as
    all-zeros for that group, which is what a model fitted without it can
    honestly say about it.
    """
    vector = []

    for name in spec.numeric:
        if name not in row:
            raise KeyError(
                f"row is missing numeric feature {name!r}; "
                f"has {sorted(row)}"
            )
        vector.append(float(row[name]))

    for name in spec.derived:
        vector.append(float(DERIVED[name](row)))

    for column in sorted(spec.categorical):
        value = row.get(column)
        vector.extend(
            1.0 if value == category else 0.0
            for category in spec.categorical[column]
        )

    return vector


def vectorize_all(rows: list[dict], spec: FeatureSpec) -> list[list[float]]:
    return [vectorize(row, spec) for row in rows]


def labels(rows: list[dict]) -> list[float]:
    """1.0 for the slot that was taken, 0.0 for the ones that were not."""
    return [1.0 if row.get("chosen") else 0.0 for row in rows]


def groups(rows: list[dict]) -> list[str]:
    """Day of each row, for grouped train/test splitting.

    Slots from one day share a calendar and are not independent. Split them
    across the boundary and the test set contains near-duplicates of training
    rows, which reports an accuracy you do not have.
    """
    return [str(row.get("day", "")) for row in rows]


def design_matrix(
    rows: list[dict], spec: FeatureSpec
) -> tuple[list[list[float]], list[float], list[str]]:
    """(X, y, groups) -- everything train.py needs from a list of rows."""
    return vectorize_all(rows, spec), labels(rows), groups(rows)


def infer_categories(rows: list[dict], column: str) -> tuple[str, ...]:
    """Sorted distinct values of a column, for building a spec at train time.

    Train time only. Sorted rather than first-seen so the same data always gives
    the same column order regardless of row order.
    """
    return tuple(sorted({str(row[column]) for row in rows if row.get(column) is not None}))


def describe(spec: FeatureSpec) -> str:
    return json.dumps(
        {"n_features": len(spec.feature_names), "features": spec.feature_names},
        indent=2,
    )
