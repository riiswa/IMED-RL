
import numpy as np
from copy import deepcopy
from Discretization import Discretization, CartpoleDiscretization, AcrobotDiscretization, NoDiscretization, CustomDisc, PendulumDisc, TaxiDisc, LakeDisc, ChainDisc, RiverDisc, SixArmsDisc, RandomGridDisc
from ActionSelector import ActionSelector, ThompsonSelector, OptimisticSelector, ThompsonSelectorTree, OptimisticSelectorTree
from V_Node import V_Node
import scipy.special
from RolloutPolicy import RandomRolloutPolicy, TaxiRolloutPolicy

class WassersteinTree:
    def __init__(self,  env_name, action_selector, amount_actions, std, C, p, file_name, maxStep, usePretrained):
        self.visit_counts = {}
        self.q_mean = {}
        self.q_std = {}
        self.values_mean = {}
        self.values_std = {}
        self.env_name = env_name
        self.idx_to_state = {}
        self.approximated = {}

        self.visited = set()
        self.global_visits = np.zeros(16)
        self.amount_actions = amount_actions
        self.init_std = std
        self.C = C
        self.dc = 0.99
        self.p = p
        self.usePretrained = usePretrained






        self.discretization = 0

        if env_name == "CartPole-v0":
            self.discretization = CartpoleDiscretization()
        elif env_name == "Acrobot-v1":
            self.discretization = AcrobotDiscretization()
        elif env_name == "Custom":
            self.discretization = CustomDisc()
        elif env_name == "Pendulum-v0":
            self.discretization = PendulumDisc()
        elif env_name == "Taxi":
            self.discretization = TaxiDisc()
        elif env_name == "FrozenLake-v1":
            self.discretization = LakeDisc()

        elif env_name == "NChain-v1":
            self.discretization = ChainDisc()
        elif env_name == "River":
            self.discretization = RiverDisc()
        elif env_name == "Arms":
            self.discretization = SixArmsDisc()
        elif env_name == "RandomGrid":
            self.discretization = RandomGridDisc()
        else:
            self.discretization = NoDiscretization()

        self.RolloutPolicy = 0
        if env_name == "Taxi":
            self.RolloutPolicy = TaxiRolloutPolicy(self.amount_actions)
        else:
            self.RolloutPolicy = RandomRolloutPolicy(self.amount_actions)

        self.setupNNPredictions(file_name)
       

        self.ActionSelector = 0
        if action_selector == "Thompson":
            self.ActionSelector = ThompsonSelector(self.q_mean, self.q_std, self.visit_counts)
        elif action_selector == "ThompsonTree":
            self.ActionSelector = ThompsonSelectorTree()
        elif action_selector == "Optimistic":
            self.ActionSelector = OptimisticSelector(self.q_mean, self.q_std, self.visit_counts, self.C)
        elif action_selector == "OptimisticTree":
            self.ActionSelector = OptimisticSelectorTree( self.C)
        elif action_selector == "OS1":
            self.ActionSelector = 2

        self.selectorString = action_selector

        self.startNode = 0
      
        self.maxStep =  maxStep


    def __len__(self):
        return len(self.q_mean)


    def clear(self):
        self.visit_counts.clear()
        self.q_mean.clear()
        self.q_std.clear()
        self.values_mean.clear()
        self.values_std.clear()

        self.idx_to_state.clear()
        self.approximated.clear()
        self.visited = set()

    def setStartV(self, initState, amount_actions):
        self.startNode = V_Node(deepcopy(self.values_mean[initState]), deepcopy(self.values_std[initState]), deepcopy(self.q_mean[initState]), deepcopy(self.q_std[initState]), initState, amount_actions)

    def updateStartV(self, state, action, amount_actions, env, done):
        newNode, _ = deepcopy(self.startNode.getNextNode(state, action, deepcopy(self.values_mean[state]), deepcopy(self.values_std[state]), deepcopy(self.q_mean[state]), deepcopy(self.q_std[state]), env, self.maxStep, self.usePretrained, self.env_name, done, amount_actions, self.RolloutPolicy))
        del self.startNode
        self.startNode = newNode


    def getActionPolicy(self, state_idx):
        mean = self.startNode.getQValueMean()
        visits = self.startNode.getStateActionVisits()
        action = np.argmax(mean)
        visit_a = np.argmax(visits)
        return action, visit_a

    def getStateIndex(self, state):
        return state, state

    def is_leaf(self, idx):
        return idx not in self.visited

    def find_leaf(self, state, env, i):
        env.seed(i+200)
        discr_state, state_idx = self.getStateIndex(state)
        
        states = []

        actions = []
        cur_state = discr_state
        values = []
        done = False

        step = 0


        currentNode = self.startNode
        states.append(currentNode)
        disc_reward = 0.0
        isLeaf = False
        while not isLeaf:
            step+= 1

            action =  self.ActionSelector.getAction(currentNode)
            actions.append(action)

            _state, r, done, _ = env.step(action)

            discr_state, state_idx = self.getStateIndex(_state)
          
            cur_state = discr_state
            currentNode, isLeaf = currentNode.TryAddQNode(state_idx, action, self.values_mean[state_idx], self.values_std[state_idx], self.q_mean[state_idx], self.q_std[state_idx], env, self.maxStep, self.usePretrained, self.env_name, done, self.amount_actions, self.RolloutPolicy)
            states.append(currentNode)
            disc_reward = disc_reward + self.dc ** step * r
           
            values.append(r)
           
           
            if done or step > self.maxStep:
                #values[-1] -= r
                break


       
        return values, cur_state, states, actions, env


    def search_batch(self, env, count, batch_size, state_init, actions, step):
        for i in range(count):
            self.search_minibatch(env, batch_size, state_init, step * count + i)

        #env.close()

    def search_minibatch(self, env, batch_size, state, i):
        backup_queue = []
       
        for cou in range(batch_size):
            values, leaf_state, states_idx_list, actions, current_env = self.find_leaf(state, deepcopy(env), i)
            if len(values) > 0:
                backup_queue.append((values, states_idx_list, actions))


        for values, state_idx_list, actions in backup_queue:
            idx = len(values) - 1

            last_node = state_idx_list[idx + 1]
            last_node.updateCombinedVisits()
            for _ in range(len(state_idx_list) - 1):
                if idx >= 0:

                    currentNode = state_idx_list[idx]
            

                    currentNode.updateStateActionVisit(actions[idx])
                    big_N = max(1, currentNode.getCombinedVisits())
                    small_n = max(1, currentNode.getStateActionVisits()[actions[idx]])
                    denom = small_n + 1
                    qmean= float((small_n * currentNode.getQValueMean()[actions[idx]] + (values[idx] + self.dc * currentNode.getSumTerm(actions[idx])))) / float(denom)

                    currentNode.setQValueMean(qmean, actions[idx])

                    q_std = float(small_n * (self.dc * currentNode.getStdSumTerm(actions[idx]))) / float(denom)
                    currentNode.setQValueStd(q_std, actions[idx])




                    #big_N = currentNode.getCombinedVisits()


                    vmean = np.power(float(sum( np.multiply(np.array(currentNode.getStateActionVisits()),np.power(np.array(currentNode.getQValueMean()), self.p))) / float(big_N)),   1 / self.p)
                    currentNode.setValueMean(vmean)

                    vstd = np.power(float(sum( np.multiply(np.array(currentNode.getStateActionVisits()), np.power(np.array(currentNode.getQValueStd()), self.p))) / float(big_N)),    1 / self.p)
                    currentNode.setValueStd(vstd)

                    currentNode.updateCombinedVisits()
                    idx -= 1




    def setupNNPredictions(self, file_name):
        file = open(file_name, 'r')
        data_strings = file.readlines()
        for i in range(len(data_strings)):
            q_string = data_strings[i]
            q_s = []
            idx1 = q_string.lstrip(' ').index("[") + 1
            for k in range(self.amount_actions):
                if k == self.amount_actions-1:
                    idx2 = q_string.lstrip(' ').index("]")
                else:
                    idx2 = q_string.lstrip(' ').index(" ") +1
                q_s.append(float(q_string[idx1:idx2]))
                q_string = q_string[idx2:].lstrip(' ')
                idx1 = 0

            pol = scipy.special.softmax(q_s)
            value = sum(np.multiply(q_s, pol))
            self.values_mean[i] = value
            self.q_mean[i] = q_s

            self.idx_to_state[i] = i
            self.visit_counts[i] = [0] * self.amount_actions
            self.q_std[i] = np.ones(self.amount_actions, dtype=np.float64) * self.init_std

            self.values_std[i] = self.init_std
        file.close()

    def saveVisitCounts(self):
        file = open(self.env_name + '_visits_wstein_' + self.selectorString + '.txt', 'a')

        file.write(str(self.global_visits))
        file.write("\n")
        file.close()