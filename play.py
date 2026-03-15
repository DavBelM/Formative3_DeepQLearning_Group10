from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

ENV_ID = "ALE/Pong-v5"
MODEL_PATH = Path("dqn_model.zip")
EPISODES = 5
SEED = 42


def make_env(seed: int):
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": "human"},
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    env = make_env(SEED)
    model = DQN.load(str(MODEL_PATH), env=env)

    for episode in range(1, EPISODES + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            total_reward += float(rewards[0])

            env.render()

        print(f"Episode {episode} total reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
