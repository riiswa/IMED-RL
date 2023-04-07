from abc import ABC, abstractmethod
import numpy as np
import itertools
class RolloutPolicy(ABC):
    def __init__(self, amount_actions):
       self.amount_actions = amount_actions
    @abstractmethod
    def getAction(self, state):
       pass


class RandomRolloutPolicy(RolloutPolicy):
    def __init__(self, amount_actions):
        super().__init__(amount_actions)


    def getAction(self, state):
        return np.random.randint(0, self.amount_actions)


class TaxiRolloutPolicy(RolloutPolicy):
    def __init__(self, amount_actions):
        super().__init__(amount_actions)
        #self.states = [0, 5, 8, 9, 10, 11, 3, 17, 18, 19, 6, 7, 2]
        self.states =  [0, 5, 8, 9, 10, 11, 3, 17, 18, 19, 6, 7, 2, 20, 21, 25, 28, 29, 30, 31, 32]
        len_init = len(self.states)
        for i in range(1, 8):
            for k in range(len_init):
                self.states.append(i * 33 + self.states[k])

    def getAction(self, state):
        if state not in self.states:
            return np.random.randint(0, self.amount_actions)
        else:
            up = 0
            down = 1
            left = 2
            right = 3
            mod_state = state % 33
            if mod_state == 0 or mod_state == 5:
                return down
            if mod_state == 8 or mod_state == 9:
                return up
            if mod_state == 3 or mod_state == 10 or mod_state == 11:
                return right
            if mod_state == 2:
                return left
            if mod_state == 6 or mod_state == 7 or mod_state == 17 or mod_state == 18 or mod_state == 19:
                if np.random.random() > 0.5:
                    return up
                else:
                    return down
            if mod_state == 20:
                if np.random.random() > 0.5:
                    return down
                else:
                    return right
            if mod_state == 21 or mod_state == 25:
                if np.random.random() > 0.5:
                    return left
                else:
                    return right
            if  mod_state == 32:
                if np.random.random() > 0.5:
                    return left
                else:
                    return up
            if mod_state == 28 or mod_state == 29 or mod_state==30 or mod_state==31:
                rnd = np.random.random()
                if rnd > 0.66666:
                    return up
                elif rnd > 0.33333:
                    return right
                else:
                    return left
