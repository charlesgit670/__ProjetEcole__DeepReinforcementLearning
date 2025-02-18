from src.visualization.log_analysis import plot_reward_length
from src.visualization.log_analysis import plot_multiple_logs
# Solo log plot
# plot_reward_length("logs/TicTacToeEnv/deep_q_learning/lr0.005_logs.json", 100)
# plot_reward_length("logs/TicTacToeEnv/double_deep_q_learning/lr0.005_logs.json", 100)
# plot_reward_length("logs/TicTacToeEnv/double_deep_q_learning_with_experience_replay/lr0.005_logs.json", 500)
# plot_reward_length("logs/TicTacToeEnv/double_deep_q_learning_with_prioritized_experience_replay/lr0.005_logs.json", 100)
# plot_reward_length("logs/TicTacToeEnv/reinforce/lr0.005_logs.json", 500)
# plot_reward_length("logs/TicTacToeEnv/reinforce_mean_baseline/lr0.005_logs.json", 500)
# plot_reward_length("logs/TicTacToeEnv/reinforce_actor_critic/lrp0.001_lrv0.001.json", 500)
# plot_reward_length("logs/TicTacToeEnv/ppo_a2c/lr0.005_logs.json", 500)

# plot_reward_lengh("logs/BalloonPOPEnv/deep_q_learning/logs_cautious_learner.json", 500)
# plot_reward_lengh("logs/BalloonPOPEnv/double_deep_q_learning/lr0.005_logs.json", 100)


# plot_reward_lengh("logs/BalloonPOPEnv/reinforce/lr0.005_logs.json", 500)

# plot_reward_lengh("logs/BalloonPOPEnv/ppo_a2c/lr0.005_logs.json", 500)

# plot_reward_length("logs/BalloonPOPEnv/double_deep_q_learning/logs.json", 1000)
plot_reward_length("logs/BalloonPOPEnv/reinforce_actor_critic/lrp0.0001_lrv0.0001.json", 1000)


# Multi logs plot
# log_paths = ['logs/BalloonPOPEnv/reinforce/lr0.001_logs.json',
#             'logs/BalloonPOPEnv/reinforce/lr0.0001_logs.json',
#             'logs/BalloonPOPEnv/reinforce/lr0.005_logs.json',
#              'logs/BalloonPOPEnv/reinforce/lr0.001_NN128_64_logs.json']
#
# plot_multiple_logs(log_paths, 1000)

# log_paths = ['logs/BalloonPOPEnv/reinforce_mean_baseline/lr0.005_logs.json',
#             'logs/BalloonPOPEnv/reinforce_mean_baseline/lr0.001_logs.json',
#             'logs/BalloonPOPEnv/reinforce_mean_baseline/lr0.0001_logs.json'
#             ]
#
# plot_multiple_logs(log_paths, 1000)

# log_paths = ["logs/BalloonPOPEnv/reinforce_actor_critic/lrp0.001_lrv0.001.json",
#             "logs/BalloonPOPEnv/reinforce_actor_critic/lrp0.0001_lrv0.001.json",
#             "logs/BalloonPOPEnv/reinforce_actor_critic/lrp0.005_lrv0.001.json",
# "logs/BalloonPOPEnv/reinforce_actor_critic/lrp0.0001_lrv0.0001.json",
# "logs/BalloonPOPEnv/reinforce_actor_critic/lrp0.0001_lrv0.005.json",
#              ]
#
# plot_multiple_logs(log_paths, 1000)

# log_paths = ["logs/BalloonPOPEnv/ppo_a2c/c20.1_epochs3_batchsize32_nsteps512_logs.json",
# "logs/BalloonPOPEnv/ppo_a2c/c20.01_epochs1_batchsize32_nsteps512_logs.json",
# "logs/BalloonPOPEnv/ppo_a2c/c20.4_epochs3_batchsize32_nsteps512_logs.json",
# "logs/BalloonPOPEnv/ppo_a2c/c20.01_epochs6_batchsize16_nsteps256_logs.json",
# "logs/BalloonPOPEnv/ppo_a2c/c20.01_epochs6_batchsize32_nsteps512_logs.json",
# "logs/BalloonPOPEnv/ppo_a2c/c20.01_epochs6_batchsize64_nsteps1024_logs.json",
# "logs/BalloonPOPEnv/ppo_a2c/c20.01_epochs12_batchsize32_nsteps512_logs.json",
# "logs/BalloonPOPEnv/ppo_a2c/c20.2_epochs3_batchsize32_nsteps512_logs.json",
#              ]
#
# plot_multiple_logs(log_paths, 1000)
