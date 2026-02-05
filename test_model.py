import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import pygame


def test_model(model_path="ppo_carracing.zip", steps=1000):
    pygame.init()

    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    # Завантажуємо модель
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    for i in range(steps):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            print("Натиснуто ESC, вихід...")
            break

        if keys[pygame.K_SPACE]:
            print("Пауза. Натисніть будь-яку клавішу щоб продовжити...")
            pygame.event.wait()

        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, info = env.step(action)

        print(f"Крок: {i}, Нагорода: {rewards[0]:.2f}")

        if dones[0]:
            print("Епізод завершено!")
            obs = env.reset()

    env.close()
    pygame.quit()


if __name__ == "__main__":
    test_model("ppo_carracing.zip", steps=2000)