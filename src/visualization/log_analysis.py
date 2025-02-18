import os
import json
import matplotlib.pyplot as plt

# def plot_reward_lengh(path_logs, mean_param=100):
#     with open(path_logs, 'r') as file:
#         logs = json.load(file)
#
#         lenght_episodes = logs["lenght_episodes"]
#         reward_episodes = logs["reward_episodes"]
#
#         mean_lenghts = [sum(lenght_episodes[i:i + mean_param]) / mean_param for i in range(0, len(lenght_episodes), mean_param) if i + mean_param <= len(lenght_episodes)]
#         mean_rewards = [sum(reward_episodes[i:i + mean_param]) / mean_param for i in range(0, len(reward_episodes), mean_param) if i + mean_param <= len(lenght_episodes)]
#
#         x_axis = [i for i in range(0, len(mean_lenghts))]
#         fig, ax = plt.subplots()
#         ax.set_title(f"mean over {mean_param} points lenghts")
#         ax.set_xlabel("Episode")
#         ax.set_ylabel("Mean lenght")
#         ax.plot(x_axis, mean_lenghts)
#
#         fig2, ax2 = plt.subplots()
#         ax2.set_title(f"mean over {mean_param} points for cumulative rewards")
#         ax2.set_xlabel("Episode")
#         ax2.set_ylabel("Mean cumulative rewards")
#         ax2.plot(x_axis, mean_rewards)
#
#         plt.show()

def plot_reward_length(path_logs, mean_param=100):
    with open(path_logs, 'r') as file:
        logs = json.load(file)

        length_episodes = logs["lenght_episodes"]
        reward_episodes = logs["reward_episodes"]

        mean_lengths = [sum(length_episodes[i:i + mean_param]) / mean_param for i in range(0, len(length_episodes), mean_param) if i + mean_param <= len(length_episodes)]
        mean_rewards = [sum(reward_episodes[i:i + mean_param]) / mean_param for i in range(0, len(reward_episodes), mean_param) if i + mean_param <= len(length_episodes)]

        x_axis = list(range(len(mean_lengths)))

        # Créer une figure avec deux sous-graphiques
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Premier graphique : Mean Lengths
        axes[0].set_title(f"Mean over {mean_param} games - Lengths")
        axes[0].plot(x_axis, mean_lengths, label="Mean Lengths", color='b')
        axes[0].set_xlabel(f"Batch of {mean_param}")
        axes[0].set_ylabel("Length")
        axes[0].legend()
        axes[0].grid()

        # Deuxième graphique : Mean Rewards
        axes[1].set_title(f"Mean over {mean_param} games - Cumulative Rewards")
        axes[1].plot(x_axis, mean_rewards, label="Mean Rewards", color='r')
        axes[1].set_xlabel(f"Batch of {mean_param}")
        axes[1].set_ylabel("Rewards")
        axes[1].legend()
        axes[1].grid()

        # Affichage des plots dans une seule fenêtre
        plt.tight_layout()
        plt.show()

def plot_multiple_logs(log_paths, mean_param=100):
    """
    Plots mean rewards and lengths for multiple log files, with the legend outside the plot.

    Args:
    - log_paths (list of str): A list of paths to the log files.
    - mean_param (int): The number of points to calculate the mean over. Default is 100.

    Returns:
    Two plots: one for the mean lengths and one for the mean rewards.
    """
    import json
    import matplotlib.pyplot as plt

    # Initialize the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    # Iterate through each log file path
    for path in log_paths:
        # Open and load the log file
        with open(path, 'r') as file:
            logs = json.load(file)

            # Extract the relevant data
            length_episodes = logs["lenght_episodes"]
            reward_episodes = logs["reward_episodes"]

            # Compute the mean lengths and rewards
            # Ensuring the correct length for the mean calculation
            mean_lengths = [sum(length_episodes[i:i + mean_param]) / mean_param
                            for i in range(0, len(length_episodes), mean_param)
                            if i + mean_param <= len(length_episodes)]
            mean_rewards = [sum(reward_episodes[i:i + mean_param]) / mean_param
                            for i in range(0, len(reward_episodes), mean_param)
                            if i + mean_param <= len(reward_episodes)]

            # Plot the means on the respective subplots
            ax1.plot(mean_lengths, label=f"{path.split('/')[-1]}")
            ax2.plot(mean_rewards, label=f"{path.split('/')[-1]}")

    # Configure the first subplot (mean lengths)
    ax1.set_title(f'Mean lengths over {mean_param} episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Mean Length')
    # Place the legend outside the first subplot
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Configure the second subplot (mean rewards)
    ax2.set_title(f'Mean cumulative rewards over {mean_param} episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Reward')
    # Place the legend outside the second subplot
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to prevent overlap and ensure space for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    # Display the plots
    plt.show()


