# rl_gold_digger
Here we have a simplified version of the gold digger game where the objective is to meet gold and avoide fire obstacles. The player controls a character that can move left, right and down using the arrow keys.

# Objective
Main goal of this mini project is to train an AI agent to play this game, the agent would need to learn how to navigate the game environment, avoid obstacles,meet gold and choose a correct action (going down) to pick the goled.

In future versions of this code gold or fires can located randomly under one of the five locations, but now for simplicity we consider these locations are not changing through training.

# Environment and Actions
The Agent can have five locations which is joint to two boolean values that states if the location is on fire or on gold or none. Combinations of location and these two boolean flags makes set of states. Also the agent has three actions in each state, left, right or down. Down action works just on fire or gold locations. When the agent moves to the same state as the gold, it receives a small reward. It also receive a small punishment when it is on fire locations. Round of game is done if agent choose down action when it is on gold (win with bigger reward) or on fire (loose with bigger punishment).
