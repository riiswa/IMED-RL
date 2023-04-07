from abc import ABC, abstractmethod
import numpy as np

class ActionSelector(ABC):
    def __init__(self, mean, std, visits):
        self.mean = mean
        self.std = std
        self.visits = visits

    @abstractmethod
    def getAction(self, state_idx):
        pass


class ThompsonSelector(ActionSelector):
    def __init__(self, mean, std, visits):
        super().__init__(mean, std, visits)


    def getAction(self, state_idx):

        mean = self.mean[state_idx]
        std = self.std[state_idx]
        high = -np.inf
        action = 0
        for i in range(len(mean)):
            val = np.random.normal(loc=mean[i], scale=std[i])

            if val > high:
                high = val
                action = i

        return action

class ThompsonSelectorTree():
    def __init__(self):
        pass

    def getAction(self, currentNode):

        mean = currentNode.getQValueMean()
        std = currentNode.getQValueStd()
        high = -np.inf
        action = 0
        for i in range(len(mean)):
            val = np.random.normal(loc=mean[i], scale=std[i])

            if val > high:
                high = val
                action = i

        return action


class OptimisticSelector(ActionSelector):
    def __init__(self, mean, std, visits, C):
        super().__init__(mean, std, visits)
        self.C = C

    def getAction(self, state_idx):

        mean = self.mean[state_idx]
        std = self.std[state_idx]
        visits = sum(self.visits[state_idx])
        if visits == 0:
            return np.random.randint(0, len(mean))
        else:
            a = mean + self.C * std *np.sqrt(np.log(visits))
            return np.argmax(a)



class OptimisticSelectorTree(ActionSelector):
    def __init__(self, C):
        self.C = C

    def getAction(self, currentNode):

        mean = currentNode.getQValueMean()
        std = currentNode.getQValueStd()
        visits = currentNode.getCombinedVisits()
        if visits == 0:
            return np.random.randint(0, len(mean))
        else:
            a = mean + self.C * std *np.sqrt(np.log(visits))
            return np.argmax(a)

class PUCTSelector(ActionSelector):
    def __init__(self, mean, std, visits, c, current_val):
        super().__init__(mean, std, visits)
        self.c = c
        self.current_val = current_val

    def getAction(self, state_idx, v_node):
        if sum(self.visits[state_idx]) == 0:
            return np.random.choice(len(self.mean))
        log_ = np.log(sum(self.visits[state_idx]))
        score = []
        for i in range(len(self.mean[state_idx])):
            score.append(self.current_val[state_idx][i] + self.c * self.mean[state_idx][i] * np.sqrt(log_ /(self.visits[state_idx][i])))
        return np.argmax(score)