import numpy as np

'''
The GoldDigger environment is a simple grid world with 5 locations, where one location can connect to gold, 
and two locations are on fire. 
The agent's goal is to navigate to the location connected to gold and pick an action to pich the gold, 
while avoiding the locations on fire. The agent can move left, right, or down.
'''

# Define the GoldDigger environment
class GoldDiggerEnvironment:
    def __init__(self):
        self.n_actions = 3
        self.n_locations = 5
        self.on_gold_location = 3
        self.on_fire_locations = [1, 5]
        self.agent_location = np.random.randint(self.n_locations)
        self.has_gold = False
        self.has_fire = False
        self.grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])
        self.reward = 0

    def step(self, action, state):
        location, has_gold, has_fire = state[0], state[1], state[2]

        if action == 0:
            location = max(location - 1, 1)
            done = False
            if location == self.on_gold_location:
                has_gold = True
                reward = 0.3
            elif location in self.on_fire_locations:
                reward = -0.5
                has_fire = True
            else:
                reward = -0.1
                has_gold = False
                has_fire = False

        elif action == 1:
            location = min(location + 1, self.n_locations )
            done = False
            if location == self.on_gold_location:
                has_gold = True
                reward = 0.3
            elif location in self.on_fire_locations:
                has_fire = True
                reward = -0.5
            else:
                reward = -0.1
                has_gold = False
                has_fire = False
        elif action == 2:
            if location == self.on_gold_location:
                reward = 10
                done = True
            elif location in self.on_fire_locations:
                reward = -5
                done = True
            else:
                reward = -0.1
                done = False
                has_gold = False
                has_fire = False

        new_state = (location, has_gold, has_fire)
        return new_state, reward, done

    def reset(self):
        self.agent_location = np.random.randint(1,self.n_locations+1)
        self.has_gold = False
        self.has_fire = False
        if self.agent_location == self.on_gold_location:
            self.has_gold = True
        if self.agent_location in self.on_fire_locations:
            self.has_fire = True
        return (self.agent_location, self.has_gold, self.has_fire)

    def render(self, agent_pos, gold_pos, fire_pos, action, done):
        grid = np.full_like(self.grid, -1)
        grid[1, gold_pos - 1] = 0
        for i in fire_pos:
            grid[1, i - 1] = 1

        if action == 2 :
            if done:
                grid[1, agent_pos-1] = 2
            else:
                grid[0,agent_pos-1] = 2
        else:
            grid[0, agent_pos - 1] = 2

        return grid

