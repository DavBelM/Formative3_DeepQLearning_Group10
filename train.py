from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# ------------------------------
# Experiment Hyperparameters
# Change these for each experiment run.
# ------------------------------
ENV_ID = "ALE/Pong-v5"
TOTAL_TIMESTEPS = 1_000_000

lr = 1e-4
gamma = 0.99
batch_size = 32
exploration_fraction = 0.1
exploration_final_eps = 0.02

SEED = 42
LOG_DIR = Path("logs")
BEST_MODEL_DIR = Path("best_model")
FINAL_MODEL_PATH = "dqn_model"


class TensorboardEpisodeStatsCallback(BaseCallback):
    """Log per-episode reward and length scalars to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.logger.record("custom/episode_reward", float(episode["r"]))
                self.logger.record("custom/episode_length", int(episode["l"]))
        return True


def make_env(seed: int):
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_env = make_env(SEED)
    eval_env = make_env(SEED + 100)

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        tensorboard_log=str(LOG_DIR),
        seed=SEED,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_MODEL_DIR),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    stats_callback = TensorboardEpisodeStatsCallback()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, stats_callback],
        tb_log_name="dqn_pong",
    )

    model.save(FINAL_MODEL_PATH)
    print("Training complete.")
    print("Best model directory: ./best_model/")
    print("Final model saved as: dqn_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
