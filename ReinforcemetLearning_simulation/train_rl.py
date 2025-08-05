import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class AircraftAvoidanceEnv(gym.Env):
    def __init__(self, num_intruders=2):
        super().__init__()
        self.dt = 1.0
        self.max_steps = 200
        self.num_intruders = num_intruders

        obs_size = 6 + 3 + num_intruders * 5  # ownship (7) + target (6) + each intruder (5)
        self.action_space = spaces.Box(low=np.array([-15, -10, -5]), high=np.array([15, 10, 5]), dtype=np.float32)
        obs_low = np.array([-10000] * obs_size)
        obs_high = np.array([10000] * obs_size)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.reset()

    def reset(self):
        self.own_pos = np.array([0.0, 0.0, 10000.0])
        self.own_heading = 0.0
        self.own_speed = 150.0
        self.own_climb = 0.0
        self.target = np.array([10000.0, 0.0, 10000.0])

        self.intruders = []
        for i in range(self.num_intruders):
            self.intruders.append({
                "pos": np.array([5000.0 + i*500, -1000.0, 10000.0]),
                "heading": 90.0,
                "speed": 150.0
            })

        self.trajectory = [self.own_pos.copy()]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = [*self.own_pos, self.own_heading, self.own_speed, self.own_climb, *self.target]
        for intr in self.intruders:
            obs.extend([*intr["pos"], intr["heading"], intr["speed"]])
        return np.array(obs, dtype=np.float32)

    def _step_intruders(self):
        for intr in self.intruders:
            rad = np.deg2rad(intr["heading"])
            dx = intr["speed"] * np.cos(rad) * self.dt
            dy = intr["speed"] * np.sin(rad) * self.dt
            intr["pos"][:2] += np.array([dx, dy])

    def _step_ownship(self, action):
        delta_heading, delta_speed, delta_climb = action
        self.own_heading += float(np.clip(delta_heading, -15, 15))
        self.own_speed += float(np.clip(delta_speed, -10, 10))
        self.own_climb += float(np.clip(delta_climb, -5, 5))
        self.own_speed = np.clip(self.own_speed, 100, 250)
        self.own_climb = np.clip(self.own_climb, -20, 20)

        rad = np.deg2rad(self.own_heading)
        dx = self.own_speed * np.cos(rad) * self.dt
        dy = self.own_speed * np.sin(rad) * self.dt
        dz = self.own_climb * self.dt
        self.own_pos += np.array([dx, dy, dz])
        self.trajectory.append(self.own_pos.copy())

    def step(self, action):
        self.steps += 1
        self._step_ownship(action)
        self._step_intruders()

        reward = -1.0
        done = self.steps >= self.max_steps
        dist_to_goal = np.linalg.norm(self.own_pos - self.target)
        if dist_to_goal < 200:
            reward += 100.0
            done = True

        for intr in self.intruders:
            dist = np.linalg.norm(self.own_pos - intr["pos"])
            if dist < 500:
                reward -= 100.0

        return self._get_obs(), reward, done, {}

# === ✅ RL Training Starts Here ===

def make_env():
    return AircraftAvoidanceEnv(num_intruders=2)

env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, verbose=1)

print("Observation shape:", env.observation_space.shape)
print("Reset observation sample shape:", env.reset().shape)

model.learn(total_timesteps=1000_000)
model.save("ppo_aircraft")
print("✅ Model training complete and saved as 'ppo_aircraft.zip'")
