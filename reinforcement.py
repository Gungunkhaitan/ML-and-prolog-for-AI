import numpy as np
import matplotlib.pyplot as plt
import random

grid_size = (5, 5)  # 5x5 grid
start = (0, 0)  # Start position
goal = (4, 4)  # Goal position
obstacles = [(2, 2), (3, 3)]  # Obstacles

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
num_episodes = 1000

actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
action_map = {0: "Right", 1: "Left", 2: "Up", 3: "Down"}

# Initialize Q-table (Grid_size x Grid_size x Actions)
Q_table = np.zeros((grid_size[0], grid_size[1], len(actions)))

# Function to take action and return next state, reward
def take_action(state, action):
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    
    # Boundary conditions
    if next_state[0] < 0 or next_state[0] >= grid_size[0] or next_state[1] < 0 or next_state[1] >= grid_size[1]:
        return state, -1  # Stay in same position, negative reward
    if next_state in obstacles:
        return state, -5
    if next_state == goal:
        return next_state, 10
    return next_state, -0.1  # Small penalty for each step

# Q-Learning Algorithm
for episode in range(num_episodes):
    state = start
    done = False
    epsilon = max(0.1, epsilon * epsilon_decay)  # Reduce exploration over time

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, len(actions) - 1)  # Explore
        else:
            action = np.argmax(Q_table[state[0], state[1]])  # Exploit
        next_state, reward = take_action(state, action)
        Q_table[state[0], state[1], action] = (1 - alpha) * Q_table[state[0], state[1], action] + \
            alpha * (reward + gamma * np.max(Q_table[next_state[0], next_state[1]]))
        state = next_state
        if state == goal:
            done = True
state = start
path = [state]
while state != goal:
    action = np.argmax(Q_table[state[0], state[1]])
    state, _ = take_action(state, action)
    path.append(state)
grid = np.zeros(grid_size)
for obs in obstacles:
    grid[obs] = -1  # Mark obstacles
grid[goal] = 2  # Mark goal
grid[start] = 1  # Mark start

plt.imshow(grid, cmap="coolwarm", interpolation="nearest")
plt.xticks(range(grid_size[1]))
plt.yticks(range(grid_size[0]))
plt.title("Grid Navigation with Q-Learning")
for i, (x, y) in enumerate(path):
    plt.text(y, x, str(i), ha="center", va="center", color="black")
plt.show()
