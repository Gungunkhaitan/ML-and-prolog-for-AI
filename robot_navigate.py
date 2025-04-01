import numpy as np
import matplotlib.pyplot as plt
import heapq
import math

# Define grid size and obstacles
grid_size = (10, 10)
obstacles = [(4, 4), (4, 5), (5, 4), (6, 6)]  # Example obstacles
start = (0, 0)
goal = (8, 8)

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def a_star_search(start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))  # Priority queue
    came_from = {}
    cost_so_far = {start: 0}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            break
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-way movement
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < grid_size[0] and 0 <= next_pos[1] < grid_size[1]) and next_pos not in obstacles:
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + euclidean_distance(next_pos, goal)
                    heapq.heappush(open_list, (priority, next_pos))
                    came_from[next_pos] = current
    
    path = []
    while goal in came_from:
        path.append(goal)
        goal = came_from[goal]
    path.reverse()
    return path

def visualize_path(path):
    grid = np.zeros(grid_size)
    for obs in obstacles:
        grid[obs] = -1  # Mark obstacles
    
    for point in path:
        grid[point] = 1  # Mark A* path
    
    plt.imshow(grid, cmap='coolwarm', origin='upper')
    plt.colorbar()
    plt.title("Robot Navigation with A*")
    plt.show()

if __name__ == "__main__":
    path = a_star_search(start, goal)
    visualize_path(path)
    print("A* Path found:", path)

