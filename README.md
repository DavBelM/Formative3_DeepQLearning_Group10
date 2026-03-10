# Formative3_DeepQLearning_Group10

DQN Atari assignment repository for Group 10.

## Group Members

- Mitali
- Caline
- Elissa

## Environment Choice

- Primary Atari environment: `ALE/Pong-v5`

You can switch to another Atari environment by passing `--env-id` in both scripts.

## Repository Files

- `train.py`: trains a DQN agent and saves model + logs.
- `play.py`: loads a trained model and runs greedy evaluation.
- `requirements.txt`: Python dependencies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Task 1: Training (train.py)

### 1) Baseline CNN run

```bash
python train.py \
	--env-id ALE/Pong-v5 \
	--policy CnnPolicy \
	--total-timesteps 500000 \
	--learning-rate 1e-4 \
	--gamma 0.99 \
	--batch-size 32 \
	--epsilon-start 1.0 \
	--epsilon-end 0.05 \
	--epsilon-fraction 0.1 \
	--experiment-name cnn_baseline \
	--model-path models/dqn_model.zip
```

### 2) Compare against MLP

```bash
python train.py \
	--env-id ALE/Pong-v5 \
	--policy MlpPolicy \
	--total-timesteps 500000 \
	--learning-rate 1e-4 \
	--gamma 0.99 \
	--batch-size 32 \
	--epsilon-start 1.0 \
	--epsilon-end 0.05 \
	--epsilon-fraction 0.1 \
	--experiment-name mlp_baseline \
	--model-path models/dqn_model_mlp.zip
```

### Training Outputs

- Final model: `models/dqn_model.zip` (or chosen output path)
- Per-run logs: `logs/<experiment_name>/`
- Episode metrics CSV: `logs/<experiment_name>/episode_metrics.csv`
- Run configuration: `logs/<experiment_name>/run_config.json`
- Global experiment tracker: `logs/experiments.csv`

## Task 2: Play/Evaluation (play.py)

```bash
python play.py \
	--env-id ALE/Pong-v5 \
	--policy CnnPolicy \
	--model-path models/dqn_model.zip \
	--episodes 3
```

Optional video recording:

```bash
python play.py \
	--env-id ALE/Pong-v5 \
	--policy CnnPolicy \
	--model-path models/dqn_model.zip \
	--episodes 2 \
	--record-video-dir videos
```

`play.py` uses `deterministic=True` in `model.predict(...)`, which is greedy action selection for DQN evaluation.

## Hyperparameter Tuning Table (Required)

Each member must complete **10 experiments** with different hyperparameter combinations.

| Member | Experiment ID | Policy    |  lr | gamma | batch_size | epsilon_start | epsilon_end | epsilon_fraction | Mean Eval Reward | Noted Behavior |
| ------ | ------------- | --------- | --: | ----: | ---------: | ------------: | ----------: | ---------------: | ---------------: | -------------- |
| Mitali | M1            | CnnPolicy |     |       |            |               |             |                  |                  |                |
| Mitali | M2            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M3            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M4            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M5            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M6            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M7            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M8            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M9            |           |     |       |            |               |             |                  |                  |                |
| Mitali | M10           |           |     |       |            |               |             |                  |                  |                |
| Caline | C1            | CnnPolicy |     |       |            |               |             |                  |                  |                |
| Caline | C2            |           |     |       |            |               |             |                  |                  |                |
| Caline | C3            |           |     |       |            |               |             |                  |                  |                |
| Caline | C4            |           |     |       |            |               |             |                  |                  |                |
| Caline | C5            |           |     |       |            |               |             |                  |                  |                |
| Caline | C6            |           |     |       |            |               |             |                  |                  |                |
| Caline | C7            |           |     |       |            |               |             |                  |                  |                |
| Caline | C8            |           |     |       |            |               |             |                  |                  |                |
| Caline | C9            |           |     |       |            |               |             |                  |                  |                |
| Caline | C10           |           |     |       |            |               |             |                  |                  |                |
| Elissa | E1            | CnnPolicy |     |       |            |               |             |                  |                  |                |
| Elissa | E2            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E3            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E4            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E5            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E6            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E7            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E8            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E9            |           |     |       |            |               |             |                  |                  |                |
| Elissa | E10           |           |     |       |            |               |             |                  |                  |                |

## Presentation Plan (10 minutes)

- 1 minute: Introduce environment + objective.
- 2 minutes: Mitali hyperparameter insights.
- 2 minutes: Caline hyperparameter insights.
- 2 minutes: Elissa hyperparameter insights.
- 2 minutes: Best configuration and policy choice (MLP vs CNN).
- 1 minute: Gameplay clip from `play.py`.

All members should be ready for Q&A on:

- Exploration vs exploitation trade-offs.
- Why specific hyperparameters helped or hurt.
- Why final model behavior makes sense.
- Why MLP or CNN was selected as final policy.

## Submission Checklist

- `train.py` committed.
- `play.py` committed.
- Best model zip file committed (or clearly accessible in repo deliverable rules).
- README includes completed hyperparameter table.
- README includes/links gameplay video showing `play.py` running.
- GitHub repository URL ready for Attempt 2.
- Zip export ready for Attempt 1.

<!-- Elissa: Added initial experiment plan E1-E5 -->

<!-- Caline: Added experiment plan C1-C5 -->

<!-- DavBelM: Refined README structure and experiment table -->

<!-- Elissa: Added experiments E6-E10 -->
