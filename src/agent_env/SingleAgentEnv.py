import numpy as np

class SingleAgentDeepEnv:
    def state_vector(self) -> np.array:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, action_id: int):
        pass

    def score(self) -> float:
        pass

    def available_actions_ids(self) -> np.ndarray:
        pass

    def available_actions_mask(self) -> np.ndarray:
        pass

    def reset(self):
        pass

    def view(self):
        pass

    def reset_random(self):
        pass
