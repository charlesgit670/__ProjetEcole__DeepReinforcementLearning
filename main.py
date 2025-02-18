from src.DRL_algorithm.DQN import deep_q_learning
from src.DRL_algorithm.DoubleDQN import double_deep_q_learning
# from src.DRL_algorithm.DoubleDQN_ExperienceReplay import double_deep_q_learning_with_experience_replay
from src.DRL_algorithm.DoubleDQN_PrioritizedExperienceReplay import double_deep_q_learning_with_prioritized_experience_replay
from src.DRL_algorithm.Reinforce import reinforce
from src.DRL_algorithm.Reinforce_actor_critic import reinforce_actor_critic
# from src.DRL_algorithm.Reinforce_mean_baseline import  reinforce_mean_baseline
from src.DRL_algorithm.Random_rollout import random_rollout_evaluation, plot_random_rollout_evaluation
from src.DRL_algorithm.ISMCTS import ISMCTS, plot_ISMCTS_evaluation
from src.DRL_algorithm.PPO_A2C import ppo_a2c
from src.DRL_algorithm.random_agent import random_evaluation

from src.agent_env.TicTacToeEnv import TicTacToeEnv
from src.agent_env.BalloonPop.BalloonPOPEnv import BalloonPOPEnv


#================================= ENV =================================
# env = TicTacToeEnv(True)
env = BalloonPOPEnv()



#================================= ALGOS =================================
# random_evaluation(env)
# ISMCTS(env, max_iteration=50)
# q_learning(env, alpha=0.3, epsilon=0.05, max_episodes_count=10000)
# deep_q_learning(env, max_episodes_count=100_000)
# double_deep_q_learning(env, lr=0.001, epsilon=0.1, max_episodes_count=250_000)
# double_deep_q_learning_with_experience_replay(env, max_episodes_count=10000)
# double_deep_q_learning_with_prioritized_experience_replay(env, max_episodes_count=10000)

# reinforce(env, max_episodes_count=100_000)
# reinforce_mean_baseline(env, max_episodes_count=10000)
# reinforce_actor_critic(env, lr_policy=0.0001, lr_value=0.0001, max_episodes_count=100_000)

# ppo_a2c(env, lr_policy=0.001, lr_value=0.001, max_episodes_count=10000)


# random_rollout_evaluation(env, max_iteration=1000, max_episodes_count=100)
# ISMCTS(env, max_iteration=1000, max_episodes_count=100)

# plot_random_rollout_evaluation(env, [10, 50, 100, 500, 1000, 2000])
plot_ISMCTS_evaluation(env, [10, 50, 100, 500, 1000, 2000])

