import pickle
from matplotlib import animation
import numpy as np
from MemoryBuffer import MemoryBuffer

import matplotlib.pyplot as plt
from matplotlib import animation

def update(env, state, action, done, ims):
    # Add frame to animation
    agent_pos =state[0]# env.agent_location
    gold_pos = env.on_gold_location
    fire_pos = env.on_fire_locations
    ims.append([plt.imshow(env.render(agent_pos, gold_pos, fire_pos, action, done), animated=True)])
    if state[1]:
        for i in range(0, 3):
            ims.append([plt.imshow(env.render(agent_pos, gold_pos, fire_pos, action, done), animated=True)])


def train_agent(env, agent, buffer_size, train_trigger_size, sample_size, NUM_EPISODES):

    # Initialize lists to store episode data
    episode_rewards = []
    mean_rewards = []
    total_rewards = [] 

    # Create memory buffer
    memory = MemoryBuffer(buffer_size)

    # Set up animation
    fig, ax = plt.subplots()
    ims = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Take action and record experience
            action = agent.act(state)
            next_state, reward, done = env.step(action, state)

            memory.add(state, action, reward, done, next_state)
            episode_reward += reward
            state = next_state

            # Train agent
            if len(memory.buffer) >= train_trigger_size:
                update(env, next_state, action, done, ims)
                states, actions, rewards, dones, next_states = memory.sample(sample_size)
                print(f"training happened at episode {episode}")
                agent.train(states, actions, rewards, dones, next_states)

        # Record episode statistics
        episode_rewards.append(episode_reward)
        total_rewards.append(sum(episode_rewards))
        mean_rewards.append(np.mean(episode_rewards))

        # Print episode statistics
        print(f"Episode {episode}: Total reward = {total_rewards[-1]}, Mean reward = {mean_rewards[-1]}")
    
    # Create animation
    anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    anim.save('./GoldDigger.gif', writer='imagemagick',fps=10)  # ,dpi= 300)

    return  episode_rewards, mean_rewards, total_rewards


def plot_metrics(episode_rewards, mean_rewards, total_rewards):
    # Clear plot
    plt.clf()

    # Plot episode reward, moving average
    #plt.plot(episode_rewards, label="Reward")
    plt.plot(mean_rewards, label="Moving Average")

    # Plot learning curve
    #plt.plot(total_rewards,
    #        label="Total Reward")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Metric")
    plt.title("Training Metrics")
    plt.savefig('reward.png')
    plt.show()

    with open("episode_reward", "wb") as fp:   #Pickling
        pickle.dump(episode_rewards, fp)
    with open("mean_rewards", "wb") as fp:   #Pickling
        pickle.dump(mean_rewards, fp)
    with open("total_rewards", "wb") as fp:   #Pickling
        pickle.dump(total_rewards, fp)  
