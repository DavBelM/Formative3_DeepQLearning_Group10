# Formative3_DeepQLearning_Group10

DQN deep reinforcement learning assignment—training agents to play Atari Pong.

## Group Members

- Mitali
- Caline
- Elissa

## The Game: Pong

We trained DQN agents on `ALE/Pong-v5` (the classic Atari Pong game). The agent controls the paddle using only the raw pixels from the screen—no hand-crafted features.

(Want to try a different Atari game? Use `--env-id` with any game like `ALE/Breakout-v5`, `ALE/SpaceInvaders-v5`, etc.)

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

Use this script to train a DQN agent on Pong from scratch.

### Basic CNN training

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

### Try MLP on raw pixels (spoiler: not recommended)

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

## Task 2: Evaluation & Video (play.py)

Load a trained model and watch it play. The script uses `deterministic=True` for greedy action selection during evaluation.

```bash
python play.py \
	--env-id ALE/Pong-v5 \
	--policy CnnPolicy \
	--model-path models/dqn_model.zip \
	--episodes 3
```

Or record gameplay video:

```bash
python play.py \
	--env-id ALE/Pong-v5 \
	--policy CnnPolicy \
	--model-path models/dqn_model.zip \
	--episodes 2 \
	--record-video-dir videos
```

## Hyperparameter Tuning Results

We ran 10 experiments per team member (30 total) with different hyperparameter combinations to see what works best for DQN on Pong.

| Member | Experiment ID | Policy    |   lr | gamma | batch_size | epsilon_start | epsilon_end | epsilon_fraction | Mean Eval Reward | Noted Behavior                                                                                               |
| ------ | ------------- | --------- | ---: | ----: | ---------: | ------------: | ----------: | ---------------: | ---------------: | ------------------------------------------------------------------------------------------------------------ |
| Mitali | M1            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |           -20.00 | Mostly stable but weak performance (agent often loses; little improvement).                                  |
| Mitali | M2            | CnnPolicy | 1e-3 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |           -21.00 | Agent not learning yet; almost always losing in this run.                                                    |
| Mitali | M3            | CnnPolicy | 1e-5 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |           -20.60 | Very low learning rate updates too slowly; only slight improvement from random play.                         |
| Mitali | M4            | CnnPolicy | 1e-4 |  0.95 |         32 |           1.0 |        0.02 |             0.10 |           -20.80 | Lower gamma reduced long-term planning; performance stayed weak with minimal learning.                       |
| Mitali | M5            | CnnPolicy | 1e-4 | 0.999 |         32 |           1.0 |        0.02 |             0.10 |           -20.40 | Higher gamma slightly improved rewards, but the agent still learned very little overall.                     |
| Mitali | M6 (200k)     | CnnPolicy | 1e-4 |  0.99 |         64 |           1.0 |        0.02 |             0.10 |           -13.33 | Enhanced with 200k timesteps instead of 50k—significantly stronger learning. Agent clearly improving at Pong. |
| Mitali | M7            | CnnPolicy | 1e-4 |  0.99 |         16 |           1.0 |        0.02 |             0.10 |           -21.00 | Small batch size made training unstable and the agent failed to learn.                                       |
| Mitali | M8            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.02 |             0.20 |           -19.80 | More exploration helped slightly, but learning remained limited overall.                                     |
| Mitali | M9            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.10 |             0.10 |           -20.40 | Higher epsilon_end still hurt exploitation, but this rerun performed slightly better than the first attempt. |
| Mitali | M10           | MlpPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.02 |             0.10 |           -21.00 | MlpPolicy performed very poorly on raw Atari frames and failed to learn useful behavior.                     |
| Caline | C1            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.01 |             0.10 |           -18.40 | Stable baseline; agent begins learning but mostly losing at 50k steps.                                       |
| Caline | C2            | CnnPolicy | 3e-4 |  0.90 |         32 |           1.0 |        0.01 |             0.10 |           -20.10 | Low gamma hurts long-term planning; agent nearly always losing.                                              |
| Caline | C3            | CnnPolicy | 3e-4 |  0.99 |        256 |           1.0 |        0.01 |             0.10 |           -17.60 | Large batch slows updates; more stable but still early learning.                                             |
| Caline | C4            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.01 |             0.30 |           -16.80 | More exploration time; agent explores longer before committing.                                              |
| Caline | C5            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.50 |             0.10 |           -20.60 | Very high epsilon_end keeps policy near-random; agent not converging.                                        |
| Caline | C6            | CnnPolicy | 1e-3 |  0.99 |        128 |           1.0 |        0.01 |             0.20 |           -15.20 | Higher lr and more exploration gave fastest improvement among Caline runs.                                   |
| Caline | C7            | CnnPolicy | 1e-5 |  0.99 |         32 |           1.0 |        0.01 |             0.10 |           -20.80 | Very low lr; network barely updates and agent learns almost nothing.                                         |
| Caline | C8            | CnnPolicy | 3e-4 |  0.99 |         32 |           1.0 |       0.001 |             0.10 |           -17.90 | Very low epsilon_end; earlier commitment gave slight improvement.                                            |
| Caline | C9            | CnnPolicy | 3e-4 |  0.97 |         64 |           1.0 |        0.05 |             0.15 |           -16.50 | Moderate gamma reduction with larger batch; agent loses less often.                                          |
| Caline | C10           | MlpPolicy | 3e-4 |  0.99 |         32 |           1.0 |        0.01 |             0.10 |           -21.00 | MLP cannot process raw pixels effectively; no learning observed.                                             |
| Elissa | E1            | CnnPolicy | 5e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.10 |           -18.80 | Baseline showed slow but consistent early learning.                                                          |
| Elissa | E2            | CnnPolicy | 5e-4 |  0.98 |         32 |           1.0 |        0.05 |             0.10 |           -19.20 | Slightly lower gamma gave marginally worse long-term behavior.                                               |
| Elissa | E3            | CnnPolicy | 5e-4 |  0.99 |        128 |           1.0 |        0.05 |             0.10 |           -17.40 | Larger batch produced steadier gradients and less noisy updates.                                             |
| Elissa | E4            | CnnPolicy | 5e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.15 |           -17.00 | More exploration helped discover better states before exploiting.                                            |
| Elissa | E5            | CnnPolicy | 2e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.10 |           -18.20 | Lower lr made learning slower but more conservative.                                                         |
| Elissa | E6            | CnnPolicy | 2e-4 |  0.95 |         64 |           1.0 |        0.05 |             0.10 |           -19.60 | Lower gamma with lower lr weakened long-term planning.                                                       |
| Elissa | E7            | CnnPolicy | 2e-4 |  0.99 |         32 |           1.0 |        0.05 |             0.25 |           -15.80 | Long exploration phase gave best Elissa result with visible improvement.                                     |
| Elissa | E8            | CnnPolicy | 1e-3 |  0.98 |         64 |           1.0 |        0.05 |             0.10 |           -16.60 | Higher lr and larger batch learned faster but with higher variance.                                          |
| Elissa | E9            | CnnPolicy | 1e-4 |  0.99 |         32 |           1.0 |        0.20 |             0.10 |           -19.80 | High epsilon_end caused persistent random actions and weak exploitation.                                     |
| Elissa | E10           | CnnPolicy | 5e-4 | 0.999 |         32 |           1.0 |        0.01 |             0.05 |           -17.20 | Very high gamma with low epsilon balanced future rewards with quicker exploitation.                          |

## Presentation Overview

This is how we organized the 10-minute presentation:

- **1 min**: Quick intro to Pong and what we were trying to do with DQN.
- **2 min**: Mitali's findings from experiments (batch size effects, learning rates, etc.).
- **2 min**: Caline's findings (exploration tuning, gamma impact).
- **2 min**: Elissa's findings (epsilon decay strategies, policy variants).
- **2 min**: Best configuration we found and why CNN beats MLP for visual inputs.
- **1 min**: Gameplay video showing the agent in action.

Key points we covered in Q&A prep:

- Why does exploration matter in DQN? How do epsilon values affect learning?
- Which hyperparameters had the biggest impact? Why did batch size help?
- Why does policy choice (CNN vs MLP) matter so much for image-based control?
- What did the agent actually learn to do in Pong?

## Deliverables

Things we included for this submission:

- ✅ `train.py` and `play.py` committed to repo.
- ✅ Best trained models saved and available.
- ✅ Complete hyperparameter results table with all 30 experiments.
- ✅ Gameplay video showing the trained agent playing Pong.
- ✅ GitHub repo ready for submission.

<!-- Elissa: Added initial experiment plan E1-E5 -->

<!-- Caline: Added experiment plan C1-C5 -->

<!-- DavBelM: Refined README structure and experiment table -->

<!-- Elissa: Added experiments E6-E10 -->

<!-- Caline: Added experiments C6-C10 -->

## Agent Gameplay

Check out the trained agent playing Pong with the best configuration (M6 enhanced):

🎮 **[Watch the agent play here](https://youtube.com/shorts/YGLDixOSNaI)**

The agent visibly improves over the baseline runs—you can see it making deliberate moves and recovering from opponent shots.

