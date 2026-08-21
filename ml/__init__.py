__version__ = "0.1.0"
__author__ = "My Mom"
__license__ = "Use it bro, I like the gym, I bet you too"

from .repository import fetch_sessions, count_training_sessions, fetch_state
from .normalize import build_training_data, clean_data
from .candidate_generator import (
    Candidate,
    duration_for,
    generate_candidates,
    load_duration_profile,
    resolve_workout,
)
from .backfill import TrainingExample, build_examples, examples_for_day
from .features import DEFAULT as DEFAULT_FEATURES
from .features import FeatureSpec, design_matrix, vectorize

__all__ = [
    "fetch_sessions",
    "count_training_sessions",
    "fetch_state",
    "build_training_data",
    "clean_data",
    "Candidate",
    "duration_for",
    "generate_candidates",
    "load_duration_profile",
    "resolve_workout",
    "TrainingExample",
    "build_examples",
    "examples_for_day",
    "DEFAULT_FEATURES",
    "FeatureSpec",
    "design_matrix",
    "vectorize",
]

