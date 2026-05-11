from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 5):
        self.size = size

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        terminated = np.array_equal(self._agent_location, self._target_location)

        truncated = False

        reward = 1 if terminated else 0
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        self.render_mode = "human"
        if self.render_mode == "human":
            for y in range(self.size - 1, -1, -1):
                row = ""
                for x in range(self.size):
                    row += " "
                    if np.array_equal([x, y], self._agent_location):
                        row += "A"
                    elif np.array_equal([x, y], self._target_location):
                        row += "T"
                    else:
                        row += "*"
                print(row)
            print()

gym.register(
    id='GridWorldEnv-v0',
    entry_point=GridWorldEnv,
    max_episode_steps=1000,
)

if __name__ == "__main__":
    env = GridWorldEnv(size=5)
    print(env.observation_space)
    print(env.action_space)
    print(env.np_random.integers(0, env.size, size=2, dtype=int))

    env = gym.make("GridWorldEnv-v0")
    obs, info = env.reset(seed=42)
    print(f"initial position agent: {obs['agent']}, target: {obs['target']}")

    test_action = [0, 1, 2, 3]
    for action in test_action:
        old_obs = obs["agent"].copy()
        obs, reward, terminated, truncated, info = env.step(action)
        new_obs = obs["agent"].copy()
        print(f'Action {action}: {old_obs} -> {new_obs}, reward: {reward}')

    env.render()