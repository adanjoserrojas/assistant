# Gym ML + CDK Plan

## Goal

Build only the new gym ML pipeline with AWS CDK.

Leave these existing resources untouched:

- Gym Command Lambda
- API Gateway
- DynamoDB table
- iPhone Shortcut
- Current Calendar Lambda
- Bedrock logic

## Repository Structure

```text
assistant/
├── ml/
│   ├── __init__.py
│   ├── repository.py
│   ├── normalize.py
│   ├── duration_profile.py
│   ├── features.py
│   ├── candidate_generator.py
│   ├── train.py
│   └── predict.py
├── ml_handlers/
│   ├── train_model_handler.py
│   └── score_candidates_handler.py
├── infrastructure/
│   ├── app.py
│   ├── cdk.json
│   ├── requirements.txt
│   └── ml_stack.py
└── tests/
    ├── test_duration_profile.py
    ├── test_features.py
    ├── test_candidates.py
    └── test_predict.py
```

## What CDK Creates

```text
GymMLStack
├── S3 bucket for model artifacts
├── Training Lambda
├── Candidate Scoring Lambda
├── EventBridge training schedule
└── IAM permissions
```

CDK references the existing DynamoDB table by name. It does not recreate or replace it.

## ML Flow

### 1. Duration Profile

- Read completed gym sessions from DynamoDB.
- Group them by workout.
- Calculate mean duration.
- Save the result to S3.

Example:

```json
{
  "Chest-Triceps": 82,
  "Back-Biceps": 76,
  "Sharms": 69
}
```

### 2. Candidate Generation

Every morning:

- Read the current workout.
- Load its mean duration.
- Find calendar windows that fit.
- Generate up to three valid candidates.

### 3. Logistic Regression

For each candidate, calculate:

- Start time
- Weekday
- Workout type
- Calendar busy minutes
- Gap before
- Gap after
- Location
- Previous attendance

The model returns attendance probability.

```text
3:00 PM → 0.52
5:30 PM → 0.84
8:00 PM → 0.61
```

The highest-scoring valid candidate wins.

## S3 Artifacts

```text
duration_profiles.json
attendance_model.joblib
model_metadata.json
```

## Build Order

### Phase 1 — Local ML Package

1. Build `repository.py`.
2. Build `normalize.py`.
3. Build `duration_profile.py`.
4. Add unit tests.
5. Generate a local duration-profile artifact.

### Phase 2 — Candidate Logic

1. Build `candidate_generator.py`.
2. Reuse the existing free-window logic.
3. Return up to three valid candidates.
4. Add tests.

### Phase 3 — Logistic Regression

1. Build `features.py`.
2. Build `train.py`.
3. Build `predict.py`.
4. Train locally with scikit-learn.
5. Save the model locally.

### Phase 4 — CDK

1. Create the CDK app.
2. Reference the existing DynamoDB table.
3. Create the S3 artifacts bucket.
4. Create both ML Lambda functions.
5. Add IAM permissions.
6. Add the EventBridge training schedule.

### Phase 5 — Deploy

```bash
cd infrastructure
cdk synth
cdk diff
cdk deploy
```

## Important Rule

Do not modify the current Calendar Lambda yet.

First prove this pipeline:

```text
Read gym data
→ calculate duration profiles
→ generate candidates
→ score candidates
→ return the winner
```

Only then connect the winner to the existing Calendar Lambda.

## Definition of Done

- Reads existing DynamoDB gym records.
- Calculates mean duration per workout.
- Saves artifacts to S3.
- Generates three valid candidates.
- Scores candidates with logistic regression.
- Returns the best candidate.
- Deploys entirely from the repository.
