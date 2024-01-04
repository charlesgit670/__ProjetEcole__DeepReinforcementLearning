from src.DRL_algorithm.DQN import deep_q_learning
from src.DRL_algorithm.DoubleDQN import double_deep_q_learning
from src.DRL_algorithm.DoubleDQN_ExperienceReplay import double_deep_q_learning_with_experience_replay
from src.DRL_algorithm.DoubleDQN_PrioritizedExperienceReplay import double_deep_q_learning_with_prioritized_experience_replay
from src.DRL_algorithm.Reinforce import reinforce
from src.agent_env.TicTacToeEnv import TicTacToeEnv

env = TicTacToeEnv(True)
# q_learning(env, alpha=0.3, epsilon=0.05, max_episodes_count=10000)
# deep_q_learning(env, max_episodes_count=100)
# double_deep_q_learning(env, max_episodes_count=100)
# double_deep_q_learning_with_experience_replay(env, max_episodes_count=1000)
# double_deep_q_learning_with_prioritized_experience_replay(env, max_episodes_count=1000)
reinforce(env, max_episodes_count=100)

