from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackAgent:
    def __init__(
            self,
            env: gym.Env,
            lr: float,
            init_eps: float,
            eps_decay: float,
            final_eps: float,
            gamma: float,
    ):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = lr
        self.gamma = gamma
        self.eps = init_eps
        self.eps_decay = eps_decay
        self.final_eps = final_eps

        self.err = []

    def get_action(self, obs):
        if np.random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = int(np.argmax(self.q_table[obs]))
        return action

    def update(
            self,
            obs,
            action,
            reward,
            next_obs,
            done):
        # print(f"obs: {obs}, action: {action}, reward: {reward}, next_obs: {next_obs}")
        future_q_value = int(not done) * np.max(self.q_table[next_obs])

        target = reward + self.gamma * future_q_value

        td_err = target - self.q_table[obs][action]

        # print(f"{self.q_table}\n")
        # print(f"{td_err}\n")

        self.q_table[obs][action] = (
            self.q_table[obs][action] + self.lr * td_err
        )
        self.err.append(td_err)

    def decay_eps(self):
        self.eps = max(self.final_eps, self.eps - self.eps_decay)

if __name__ == "__main__":
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    eps_decay = start_epsilon / n_episodes / 2
    final_epsilon = 0.1

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)

    agent = BlackjackAgent(
        env=env,
        lr=learning_rate,
        init_eps=start_epsilon,
        eps_decay=eps_decay,
        final_eps=final_epsilon,
        gamma=0.95,
    )

    from tqdm import tqdm

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)

            done = terminated or truncated
            obs = next_obs

        agent.decay_eps()

    from matplotlib import pyplot as plt


    def get_moving_avgs(arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window


    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.err,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


    # Test the trained agent
    def test_agent(agent, env, num_episodes=1000):
        """Test agent performance without learning or exploration."""
        total_rewards = []

        # Temporarily disable exploration for testing
        old_epsilon = agent.eps
        agent.eps = 0.0  # Pure exploitation

        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        # Restore original epsilon
        agent.eps = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)

        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")


    # Test your agent
    test_agent(agent, env)