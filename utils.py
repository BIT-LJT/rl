import matplotlib.pyplot as plt
import numpy as np
import os

def render_env(env, save=False, filename=None):
    """
    Render the environment: plot agents, resident points, and central station.
    """
    plt.figure(figsize=(8, 8))

    # Plot resident points
    for idx, point in enumerate(env.points):
        if env.done_points[idx] == 1:
            plt.scatter(point[0], point[1], c='green', marker='o')  # Completed
        else:
            plt.scatter(point[0], point[1], c='red', marker='x')  # Pending

    # Plot agents
    for pos in env.agent_positions:
        plt.scatter(pos[0], pos[1], c='blue', marker='s')

    # Plot central detection station
    plt.scatter(env.central_station[0], env.central_station[1], c='black', marker='*', s=200)

    plt.xlim(0, env.size)
    plt.ylim(0, env.size)
    plt.title(f"Sampling Task - Time {env.time/60:.1f} minutes")
    plt.grid(True)

    if save and filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()


def plot_rewards(all_episode_rewards, filename=None):
    """
    Plot training reward curves for all agents.
    """
    if not all_episode_rewards:
        return
        
    all_episode_rewards_by_agent = list(zip(*all_episode_rewards))
    plt.figure(figsize=(10, 6))

    for idx, rewards in enumerate(all_episode_rewards_by_agent):
        plt.plot(rewards, label=f'Agent {idx+1}')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curves')
    plt.legend()
    plt.grid(True)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()



def plot_reward_curve2(all_rewards, filename="reward_curve2.png", smooth=0.9):
    """
    all_rewards: List of list，每个元素是某一轮中所有 agent 的总 reward
    smooth: 平滑系数，0.9 比较平滑，0.0 完全原始
    """
    if not all_rewards:
        return

    all_rewards = np.array(all_rewards)
    if all_rewards.ndim < 2:
        return

    smoothed = []
    for i in range(all_rewards.shape[1]):
        rewards = all_rewards[:, i]
        if len(rewards) == 0:
            continue
        smoothed_rewards = [rewards[0]]
        for r in rewards[1:]:
            smoothed_rewards.append(smoothed_rewards[-1] * smooth + r * (1 - smooth))
        smoothed.append(smoothed_rewards)

    plt.figure(figsize=(10, 6))
    for i, rewards in enumerate(smoothed):
        plt.plot(rewards, label=f"Agent {i+1}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Smoothed)")
    plt.title("Agent Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    plt.close()



def plot_agent_trajectories(agent_paths, points, central_station, episode_id, foldername="trajectories"):
    """
    agent_paths: list of list of positions per agent, shape: [num_agents][timesteps][2]
    points: ndarray of shape [num_points, 2]
    central_station: [2] array
    """
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for idx, path in enumerate(agent_paths):
        # FIX: Add a safety check to ensure the path is not empty and is a valid 2D array.
        if not path or len(path) < 2:
            continue
        
        path_array = np.array(path)
        
        if path_array.ndim != 2 or path_array.shape[1] != 2:
            print(f"Warning: Skipping plotting for agent {idx+1} due to invalid path shape: {path_array.shape}")
            continue

        plt.plot(path_array[:, 0], path_array[:, 1], color=colors[idx % len(colors)], label=f"Agent {idx+1}", marker='.')
        plt.scatter(path_array[0, 0], path_array[0, 1], color=colors[idx % len(colors)], marker='o', s=100)
        plt.scatter(path_array[-1, 0], path_array[-1, 1], color=colors[idx % len(colors)], marker='x', s=100)

    plt.scatter(points[:, 0], points[:, 1], color='gray', marker='^', label='Sampling Points')
    plt.scatter(central_station[0], central_station[1], color='black', marker='s', s=150, label='Central Station')

    plt.title(f"Agent Trajectories at Episode {episode_id}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs(foldername, exist_ok=True)
    filename = os.path.join(foldername, f"episode_{episode_id}_trajectories.png")
    plt.savefig(filename)
    plt.close()
