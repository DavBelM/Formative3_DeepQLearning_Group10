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

| Member | Experiment ID | Policy    |   lr | gamma | batch_size | epsilon_start | epsilon_end | epsilon_fraction | Mean Eval Reward | Noted Behavior |
| ------ | ------------- | --------- | ---: | ----: | ---------: | ------------: | ----------: | ---------------: | ---------------: | -------------- |
| Mitali | M1            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |           -20.00 | Mostly stable but weak performance (agent often loses; little improvement). |
| Mitali | M2            | CnnPolicy | 1e-3 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |           -21.00 | Agent not learning yet; almost always losing in this run. |
| Mitali | M3            | CnnPolicy | 1e-5 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |                  |                |
| Mitali | M4            | CnnPolicy | 1e-4 |  0.95 |         32 |           1.0 |        0.02 |             0.10 |                  |                |
| Mitali | M5            | CnnPolicy | 1e-4 | 0.999 |         32 |           1.0 |        0.02 |             0.10 |                  |                |
| Mitali | M6            | CnnPolicy | 1e-4 |  0.99 |         64 |           1.0 |        0.02 |             0.10 |                  |                |
| Mitali | M7            | CnnPolicy | 1e-4 |  0.99 |         16 |           1.0 |        0.02 |             0.10 |                  |                |
| Mitali | M8            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.02 |             0.20 |                  |                |
| Mitali | M9            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.10 |             0.10 |                  |                |
| Mitali | M10           | MlpPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |                  |                |
| Caline | C1            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.01 |             0.10 |           -18.40 | Stable baseline; agent begins learning but mostly losing at 50k steps. |
| Caline | C2            | CnnPolicy | 3e-4 |  0.90 |         32 |           1.0 |        0.01 |             0.10 |           -20.10 | Low gamma hurts long-term planning; agent nearly always losing. |
| Caline | C3            | CnnPolicy | 3e-4 |  0.99 |        256 |           1.0 |        0.01 |             0.10 |           -17.60 | Large batch slows updates; more stable but still early learning. |
| Caline | C4            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.01 |             0.30 |           -16.80 | More exploration time; agent explores longer before committing. |
| Caline | C5            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.50 |             0.10 |           -20.60 | Very high epsilon_end keeps policy near-random; agent not converging. |
| Caline | C6            | CnnPolicy | 1e-3 |  0.99 |        128 |           1.0 |        0.01 |             0.20 |           -15.20 | Higher lr and more exploration gave fastest improvement among Caline runs. |
| Caline | C7            | CnnPolicy | 1e-5 |  0.99 |         32 |           1.0 |        0.01 |             0.10 |           -20.80 | Very low lr; network barely updates and agent learns almost nothing. |
| Caline | C8            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |       0.001 |             0.10 |           -17.90 | Very low epsilon_end; earlier commitment gave slight improvement. |
| Caline | C9            | CnnPolicy | 3e-4 |  0.97 |         64 |           1.0 |        0.05 |             0.15 |           -16.50 | Moderate gamma reduction with larger batch; agent loses less often. |
| Caline | C10           | MlpPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.01 |             0.10 |           -21.00 | MLP cannot process raw pixels effectively; no learning observed. |
| Elissa | E1            | CnnPolicy | 5e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.10 |           -18.80 | Baseline showed slow but consistent early learning. |
| Elissa | E2            | CnnPolicy | 5e-4 |  0.98 |         32 |           1.0 |        0.05 |             0.10 |           -19.20 | Slightly lower gamma gave marginally worse long-term behavior. |
| Elissa | E3            | CnnPolicy | 5e-4 |  0.99 |        128 |           1.0 |        0.05 |             0.10 |           -17.40 | Larger batch produced steadier gradients and less noisy updates. |
| Elissa | E4            | CnnPolicy | 5e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.15 |           -17.00 | More exploration helped discover better states before exploiting. |
| Elissa | E5            | CnnPolicy | 2e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.10 |           -18.20 | Lower lr made learning slower but more conservative. |
| Elissa | E6            | CnnPolicy | 2e-4 |  0.95 |         64 |           1.0 |        0.05 |             0.10 |           -19.60 | Lower gamma with lower lr weakened long-term planning. |
| Elissa | E7            | CnnPolicy | 2e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.25 |           -15.80 | Long exploration phase gave best Elissa result with visible improvement. |
| Elissa | E8            | CnnPolicy | 1e-3 |  0.98 |         64 |           1.0 |        0.05 |             0.10 |           -16.60 | Higher lr and larger batch learned faster but with higher variance. |
| Elissa | E9            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.20 |             0.10 |           -19.80 | High epsilon_end caused persistent random actions and weak exploitation. |
| Elissa | E10           | CnnPolicy | 5e-4 | 0.999 |         32 |           1.0 |        0.01 |             0.05 |           -17.20 | Very high gamma with low epsilon balanced future rewards with quicker exploitation. |

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

<!-- Caline: Added experiments C6-C10 -->
