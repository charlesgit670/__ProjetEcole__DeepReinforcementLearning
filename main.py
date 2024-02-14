from src.DRL_algorithm.DQN import deep_q_learning
from src.DRL_algorithm.DoubleDQN import double_deep_q_learning
# from src.DRL_algorithm.DoubleDQN_ExperienceReplay import double_deep_q_learning_with_experience_replay
# from src.DRL_algorithm.DoubleDQN_PrioritizedExperienceReplay import double_deep_q_learning_with_prioritized_experience_replay
from src.DRL_algorithm.Reinforce import reinforce
# from src.DRL_algorithm.Reinforce_actor_critic import reinforce_actor_critic
# from src.DRL_algorithm.Reinforce_mean_baseline import  reinforce_mean_baseline
from src.DRL_algorithm.Random_rollout import random_rollout
from src.DRL_algorithm.ISMCTS import ISMCTS
from src.DRL_algorithm.PPO_A2C import ppo_a2c

from src.agent_env.TicTacToeEnv import TicTacToeEnv
from src.agent_env.BalloonPop.main import BalloonPOPEnv

#================================= ENV =================================
# env = TicTacToeEnv(True)
env = BalloonPOPEnv()


#================================= ALGOS =================================
# q_learning(env, alpha=0.3, epsilon=0.05, max_episodes_count=10000)
deep_q_learning(env, max_episodes_count=100_000)
# double_deep_q_learning(env, max_episodes_count=10000)
# double_deep_q_learning_with_experience_replay(env, max_episodes_count=10000)
# double_deep_q_learning_with_prioritized_experience_replay(env, max_episodes_count=10000)

# reinforce(env, max_episodes_count=100_000)
# reinforce_mean_baseline(env, max_episodes_count=10000)
# reinforce_actor_critic(env, max_episodes_count=10000)

# ppo_a2c(env, lr_policy=0.01, lr_value=0.01, max_episodes_count=100_000)

# random_rollout(env, max_time=10000)
# ISMCTS(env, time_per_action=0.01, max_episodes_count=10000)



#================================= ALGOS BY TIME =================================
