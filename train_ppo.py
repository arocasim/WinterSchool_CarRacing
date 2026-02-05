import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class WatchCallback(BaseCallback):
    def __init__(self, every_steps=10_000, demo_steps=300, verbose=0):
        super().__init__(verbose)
        self.every_steps = every_steps
        self.demo_steps = demo_steps
        self.next_watch = every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_watch:
            self.next_watch += self.every_steps
            self._watch_demo()
        return True

    def _watch_demo(self):
        import pygame
        pygame.init()

        env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4)

        obs = env.reset()
        for _ in range(self.demo_steps):
            pygame.event.pump()
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                break

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        env.close()
        pygame.quit()


def main():
    env = make_vec_env("CarRacing-v3", n_envs=1, env_kwargs={"continuous": True})
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        clip_range=0.2,
    )

    checkpoint = CheckpointCallback(save_freq=10_000, save_path="checkpoints", name_prefix="ppo")
    watch = WatchCallback(every_steps=10_000, demo_steps=300)

    model.learn(total_timesteps=1_000_000, callback=[checkpoint, watch])
    model.save("ppo_carracing")

    env.close()


if __name__ == "__main__":
    main()
