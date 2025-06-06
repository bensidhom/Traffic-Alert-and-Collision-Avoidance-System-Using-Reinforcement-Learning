import streamlit as st
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import plotly.graph_objects as go
import time

class AircraftAvoidanceEnv(gym.Env):
    def __init__(self, num_intruders=2, start_pos=(0.0, 0.0, 10000.0)):
        super().__init__()
        self.dt = 1.0
        self.max_steps = 3600
        self.num_intruders = num_intruders
        self.start_pos = np.array(start_pos, dtype=np.float64)

        obs_size = 6 + 3 + num_intruders * 5
        self.action_space = spaces.Box(low=np.array([-15, -10, -5]), high=np.array([15, 10, 5]), dtype=np.float32)
        obs_low = np.array([-10000] * obs_size, dtype=np.float32)
        obs_high = np.array([10000] * obs_size, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.reset()

    def reset(self):
        self.own_pos = self.start_pos.copy()
        self.own_heading = 0.0
        self.own_speed = 150.0
        self.own_climb = 0.0
        self.target = np.array([10000.0, 0.0, 10000.0], dtype=np.float64)

        self.intruders = []
        for i in range(self.num_intruders):
            self.intruders.append({
                "pos": np.array([5000.0 + i*500, -1000.0, 10000.0], dtype=np.float64),
                "heading": 90.0,
                "speed": 150.0
            })

        self.trajectory = [self.own_pos.copy()]
        self.intruder_trajectories = [[intr["pos"].copy()] for intr in self.intruders]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = [*self.own_pos, self.own_heading, self.own_speed, self.own_climb, *self.target]
        for intr in self.intruders:
            obs.extend([*intr["pos"], intr["heading"], intr["speed"]])
        return np.array(obs, dtype=np.float32)

    def _step_intruders(self):
        for i, intr in enumerate(self.intruders):
            intr["heading"] += np.random.uniform(-10, 10)
            rad = np.deg2rad(intr["heading"])
            dx = intr["speed"] * np.cos(rad) * self.dt
            dy = intr["speed"] * np.sin(rad) * self.dt
            delta = np.array([dx, dy], dtype=np.float64)
            intr["pos"][:2] += delta
            self.intruder_trajectories[i].append(intr["pos"].copy())

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
        self.own_pos += np.array([dx, dy, dz], dtype=np.float64)
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

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🛫 RL Aircraft Real-Time Simulation (Animated Intruders)")

num_intruders = st.sidebar.slider("Number of Intruders", 1, 5, 2)
x_start = st.sidebar.slider("Start X", -1000, 1000, 0)
y_start = st.sidebar.slider("Start Y", -1000, 1000, 0)
z_start = st.sidebar.slider("Start Altitude (Z)", 5000, 15000, 10000)

model = PPO.load(r"C:\all\se\cpa\ppo_aircraft.zip")
vec_env = DummyVecEnv([lambda: AircraftAvoidanceEnv(num_intruders=num_intruders, start_pos=(x_start, y_start, z_start))])
obs = vec_env.reset()

raw_env = vec_env.envs[0]
traj = np.array(raw_env.trajectory)
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode='lines+markers', name='Agent Path'))

for i, intr_path in enumerate(raw_env.intruder_trajectories):
    path = np.array(intr_path)
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines+markers', name=f"Intruder {i+1}"))

fig.add_trace(go.Scatter3d(x=[raw_env.target[0]], y=[raw_env.target[1]], z=[raw_env.target[2]], mode='markers', marker=dict(size=6, color='green'), name='Target'))

fig.update_layout(title="Aircraft Avoidance Simulation (Zoom stays fixed)", scene=dict(
    xaxis=dict(title='X', range=[-2000, 20000]),
    yaxis=dict(title='Y', range=[-5000, 5000]),
    zaxis=dict(title='Altitude (Z)', range=[5000, 15000])
))

plot = st.plotly_chart(fig, use_container_width=True)

for step in range(3600):
    time.sleep(2)
    action, _ = model.predict(obs)
    obs, reward, done, _ = vec_env.step(action)

    traj = np.array(raw_env.trajectory)
    fig.data[0].x = traj[:, 0]
    fig.data[0].y = traj[:, 1]
    fig.data[0].z = traj[:, 2]

    for i, intr_path in enumerate(raw_env.intruder_trajectories):
        path = np.array(intr_path)
        fig.data[i+1].x = path[:, 0]
        fig.data[i+1].y = path[:, 1]
        fig.data[i+1].z = path[:, 2]

    plot.plotly_chart(fig, use_container_width=True)

    if done:
        st.success("✅ Target reached or max steps completed!")
        break
