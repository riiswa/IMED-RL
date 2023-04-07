import math
import numpy as np
from copy import deepcopy

from scipy.optimize import minimize_scalar

from Discretization import Discretization, CartpoleDiscretization, AcrobotDiscretization, NoDiscretization, CustomDisc, PendulumDisc, TaxiDisc, LakeDisc, ChainDisc, RiverDisc, SixArmsDisc, RandomGridDisc
from ActionSelector import ActionSelector, ThompsonSelector, OptimisticSelector, ThompsonSelectorTree
from V_Node import V_Node_PUCT
import scipy.special
import random
from RolloutPolicy import RandomRolloutPolicy, TaxiRolloutPolicy

import torch.nn as nn
import torch

class PUCTMCTS_Tree:
    def __init__(self,  env_name, action_selector, amount_actions, std, C,p, file_name, maxStep, usePretrained):
        self.visit_counts = {}
        self.qvalues = {}
        self.env_name = env_name
        self.values = {}

        self.idx_to_state = {}
        self.approximated = {}

        self.visited = set()
        self.global_visits = np.zeros(16)
        self.amount_actions = amount_actions
        self.init_std = std
        
        self.dc = 0.99
        self.p = p
        self.c = C


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
        elif env_name == "Taxi-v3":
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

        elif action_selector == "OS1":
            self.ActionSelector = 2

        self.startNode = 0
      
        self.maxStep = maxStep



    def __len__(self):
        return len(self.qvalues)


    def clear(self):
        self.visit_counts.clear()
        self.qvalues.clear()
        self.values.clear()


        self.idx_to_state.clear()
        self.approximated.clear()
        self.visited = set()

    def setStartV(self, initState, amount_actions):

        self.startNode = V_Node_PUCT(deepcopy(self.values[initState]), deepcopy(self.qvalues[initState]), initState, amount_actions)

    def updateStartV(self, state, action, amount_actions, env, done):

        newNode, _ = deepcopy(self.startNode.getNextNode(state, action, deepcopy(self.values[state]), deepcopy(self.qvalues[state]), env, self.maxStep, self.usePretrained, self.env_name, done, amount_actions, self.RolloutPolicy))

        del self.startNode
        self.startNode = newNode



    def getActionPolicy(self, state_idx):
        mean = self.startNode.getQValue()
        visits = self.startNode.getStateActionVisits()
        action = np.argmax(mean)
        visit_a = np.argmax(visits)
        return action, visit_a

    def getStateIndex(self, state):
        return state, state


    def is_leaf(self, idx):
        return idx not in self.visited

    def compute_scores(self, currentNode):
        log_ = np.log(currentNode.getCombinedVisits())
        scores = []
        for i in range(self.amount_actions):
            s_a_visits = max(1.0, float(currentNode.getStateActionVisits()[i]))
            scores.append(currentNode.getQValue()[i] + self.c * np.sqrt(log_ / (s_a_visits)))

        return scores


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
            step += 1

            if currentNode.getCombinedVisits() == 0:
                action = np.random.choice(self.amount_actions)
            else:
                action = np.argmax(self.compute_scores(currentNode))
            prev_state = cur_state
            _state, r, done, _ = env.step(action)

            discr_state, state_idx = self.getStateIndex(_state)
           
            cur_state = discr_state
            currentNode, isLeaf = currentNode.TryAddQNode(state_idx, action, self.values[state_idx], self.qvalues[state_idx], env, self.maxStep, self.usePretrained, self.env_name, done, self.amount_actions, self.RolloutPolicy)


            states.append(currentNode)
            actions.append(action)
            disc_reward = disc_reward + self.dc ** step * r

            values.append(r)

            if self.env_name == "Arms" and r > 0:
                done = True
            if done or step > self.maxStep:
                #values[-1] -= r
                break


        #if currentNode.isLeaf() and done != True:
            #leaves = self.LeafAdder.getLeaves(currentNode.getState())
           # currentNode.AddLeaves(leaves)

        """
        discr_state, state_idx = self.getStateIndex(cur_state)
        action = self.getActionOS(state_idx)
        actions.append(action)
        """
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

                    v_val = state_idx_list[idx]
                    v_val.updateStateActionVisit(actions[idx])
                    vis = max(1, v_val.getStateActionVisits()[actions[idx]])

                    new_q_val = float((values[idx] + self.dc * v_val.getSumTerm(actions[idx]))) / float(vis + 1)
                    v_val.setQValue(new_q_val, actions[idx])
                  #  v_val.updateStateActionVisit(actions[idx])



                    vis = max(1, v_val.getCombinedVisits())
                    s_a_visits = v_val.getStateActionVisits()
                    new_val = np.power(float( sum(np.multiply(np.array(s_a_visits), np.power(np.array(v_val.getQValue()), self.p)))) / float(vis), 1.0 / self.p)
                    v_val.setValue(new_val)
                    v_val.updateCombinedVisits()




                    idx -= 1



    def setupNNPredictions(self, file_name):

        file = open(file_name, 'r')
        data_strings = file.readlines()
        for i in range(len(data_strings)):
            q_string = data_strings[i]
            q_s = []
            idx1 = q_string.lstrip(' ').index("[") + 1
            for k in range(self.amount_actions):
                if k == self.amount_actions - 1:
                    idx2 = q_string.lstrip(' ').index("]")
                else:
                    idx2 = q_string.lstrip(' ').index(" ") + 1
                q_s.append(float(q_string[idx1:idx2]))
                q_string = q_string[idx2:].lstrip(' ')
                idx1 = 0

            pol = scipy.special.softmax(q_s)
            value = sum(np.multiply(q_s, pol))
            self.values[i] = value
            self.qvalues[i] = q_s

            self.idx_to_state[i] = i
            self.visit_counts[i] = [0] * self.amount_actions

        file.close()



    def saveVisitCounts(self):
        file = open(self.env_name + '_visits_puct.txt', 'a')

        file.write(str(self.global_visits))
        file.write("\n")
        file.close()

    # def compute_scores(self, currentNode):
    #     log_ = np.log(currentNode.getCombinedVisits())
    #     scores = []
    #     for i in range(self.amount_actions):
    #         s_a_visits = max(1.0, float(currentNode.getStateActionVisits()[i]))
    #         scores.append(currentNode.getQValue()[i] + self.c * np.sqrt(log_ / (s_a_visits)))
    #
    #     return scores


class KInf(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(0.))

    def forward(self, F, mu):
        return -torch.sum(torch.log(1 - (F - mu) * self.x))


class IMEDMCTS_Tree(PUCTMCTS_Tree):
    def __init__(self, env_name, action_selector, amount_actions, std, C, p, file_name, maxStep, usePretrained):
        super().__init__(env_name, action_selector, amount_actions, std, C, p, file_name, maxStep, usePretrained)

    def compute_scores(self, currentNode):
        best_mean = max(currentNode.getQValue())
        scores = []

        for i in range(self.amount_actions):
            res = minimize_scalar(lambda x: - np.sum(np.log(1 - (currentNode.getRewards(i) - best_mean) * x)), bounds=(0, 1 / (1 - best_mean - 1e-2)), method='bounded')
            x = -res.fun
            s_a_visits = max(1.0, float(currentNode.getStateActionVisits()[i]))
            scores.append(-(s_a_visits * x.item() + np.log(s_a_visits)))

        return scores

