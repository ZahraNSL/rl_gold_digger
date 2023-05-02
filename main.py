from GoldDiggerAgent import A2C
from GoldDiggerEnv import GoldDiggerEnvironment

from train_utils import train_agent, plot_metrics

# Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
buffer_size = 1
NUM_EPISODES = 5000
train_trigger_size = 1
sample_size = 1

# Create the environment and agent
env = GoldDiggerEnvironment()
agent = A2C(state_size=3, action_size=env.n_actions, learning_rate=LEARNING_RATE, gamma=DISCOUNT_FACTOR)

episode_rewards, mean_rewards, total_rewards = train_agent(env, agent, buffer_size, train_trigger_size, sample_size, NUM_EPISODES)
plot_metrics(episode_rewards,mean_rewards,total_rewards)



