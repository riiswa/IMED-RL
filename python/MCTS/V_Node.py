from copy import deepcopy
import numpy as np
class V_Node:
    def __init__(self, v_mean, v_std, q_mean, q_std, state, amount_of_actions):

        self.combined_visits = 0
        self.visits = [0] * amount_of_actions
        self.state = state
        self.visited_q_nodes = []
        for i in range(amount_of_actions):
            self.visited_q_nodes.append({})
        self.amount_of_actions = amount_of_actions

        self.value_mean = v_mean
        self.value_std = v_std
        self.q_mean = q_mean
        self.q_std = q_std

        self.usePretrained = True

    def __del__(self):
        for i in range(self.amount_of_actions):
            self.visited_q_nodes[i].clear()

    def getCombinedVisits(self):
        return self.combined_visits
    def getStateActionVisits(self):
        return self.visits
    def getValueMean(self):
        return self.value_mean
    def getValueStd(self):
        return self.value_std
    def getQValueMean(self):
        return self.q_mean
    def getQValueStd(self):
        return self.q_std

    def setValueMean(self, newval):
        self.value_mean = newval
    def setValueStd(self, newval):
        self.value_std = newval
    def setQValueMean(self, newval, action):
        self.q_mean[action] = newval
    def setQValueStd(self, newval,action):
        self.q_std[action] = newval

    def updateCombinedVisits(self):
        self.combined_visits+=1
    def updateStateActionVisit(self, action):
        self.visits[action] +=1

    def getState(self):
        return self.state

    def getNextNode(self, state, action,  v_mean, v_std, q_mean, q_std, env, maxSteps, usePretrained, env_name, done, amount_actions, RolloutPolicy):
        return self.TryAddQNode(state, action, v_mean, v_std, q_mean, q_std, env, maxSteps, usePretrained, env_name, done, amount_actions, RolloutPolicy)

    def isLeaf(self):
        #if len(self.visited_q_nodes[0]) == 0:
            #return True
        if self.combined_visits == 0:
            return True
        return False

    def TryAddQNode(self, state, action, v_mean, v_std, q_mean, q_std, env, maxSteps, usePretrained, env_name, done_flag, amount_actions, RolloutPolicy):
        isLeaf = False
        if state not in self.visited_q_nodes[action]:
            self.visited_q_nodes[action][state] = V_Node(v_mean, v_std, q_mean, q_std, state, amount_actions)
            isLeaf = True
        if not usePretrained and not done_flag:
            c_env = deepcopy(env)
            disc_rew = 0.0
            in_state = 0
            for step in range(maxSteps):
                act = RolloutPolicy.getAction(in_state)
                if env_name == "Arms":
                    if in_state != 0:
                        act = in_state - 1
                in_state, r, done, _ = c_env.step(act)
                disc_rew = disc_rew + 0.99 ** step * r
                if env_name == "Arms" and r > 0:
                    done = True
                if done:
                    break
            self.visited_q_nodes[action][state].setValueMean(disc_rew)
        return self.visited_q_nodes[action][state], isLeaf

    def AddLeaves(self, leaves):
        for act in range(len(leaves)):
            for leave in leaves[act]:
                self.visited_q_nodes[act][leave.getState()] = deepcopy(leave)

    def getSumTerm(self, action):
        
      
        sum = 0.0
        comb_visits = 0.0

        for key, Node in self.visited_q_nodes[action].items():
            #visits = max(1, Node.getCombinedVisits())
            
            visits = Node.getCombinedVisits()
            comb_visits += visits
            sum+= visits * Node.getValueMean()
            
      
        return sum/comb_visits
      

    def getStdSumTerm(self, action):
        sum = 0.0
        comb_visits = 0.0

        for key, Node in self.visited_q_nodes[action].items():
            #visits = max(1, Node.getCombinedVisits())
            visits = Node.getCombinedVisits()
            comb_visits += visits
            sum+= visits * Node.getValueStd()

        return sum / comb_visits


class V_Node_PUCT:
    def __init__(self, value, qvalue, state, amount_of_actions):

        self.combined_visits = 0

        self.visits = [0] * amount_of_actions
        self.state = state
        self.visited_q_nodes = []
        for i in range(amount_of_actions):
            self.visited_q_nodes.append({})
        self.amount_of_actions = amount_of_actions

        self.value = value

        self.qvalue = qvalue


    def __del__(self):
        for i in range(self.amount_of_actions):
            self.visited_q_nodes[i].clear()

    def getCombinedVisits(self):
        return self.combined_visits
    def getStateActionVisits(self):
        return self.visits
    def getValue(self):
        return self.value

    def getQValue(self):
        return self.qvalue


    def setValue(self, newval):
        self.value = newval

    def setQValue(self, newval, action):
        self.qvalue[action] = newval


    def updateCombinedVisits(self):
        self.combined_visits+=1
    def updateStateActionVisit(self, action):
        self.visits[action] +=1

    def getState(self):
        return self.state

    def getNextNode(self, state, action,  value, qvalue, env, maxSteps, usePretrained, env_name, done, amount_actions, RolloutPolicy):
        return self.TryAddQNode(state, action, value, qvalue, env, maxSteps, usePretrained, env_name, done, amount_actions, RolloutPolicy)

    def isLeaf(self):
        if len(self.visited_q_nodes[0]) == 0:
            return True
        return False

    def TryAddQNode(self, state, action, value, qvalue, env, maxSteps, usePretrained, env_name, done_flag, amount_actions, RolloutPolicy):
        isLeaf = False
        if state not in self.visited_q_nodes[action]:
            self.visited_q_nodes[action][state] = V_Node_PUCT(value, qvalue, state, amount_actions)
            isLeaf = True
        if not usePretrained and not done_flag:
            c_env = deepcopy(env)
            disc_rew = 0.0
            in_state = state
            for step in range(maxSteps):
                act = RolloutPolicy.getAction(in_state)
                if env_name == "Arms":
                    if in_state != 0:
                        act = in_state - 1
                in_state, r, done, _ = c_env.step(act)
                disc_rew = disc_rew + 0.99 ** step * r

                if env_name == "Arms" and r > 0:
                    done = True
                if done:
                    break
          
            self.visited_q_nodes[action][state].setValue(disc_rew)
        return self.visited_q_nodes[action][state], isLeaf

    def AddLeaves(self, leaves):
        for act in range(len(leaves)):
            for leave in leaves[act]:
                self.visited_q_nodes[act][leave.getState()] = deepcopy(leave)

    def AddLeavesRollout(self, leaves, env, dc):
        for act in range(len(leaves)):
            for leave in leaves[act]:
                self.visited_q_nodes[act][leave.getState()] = deepcopy(leave)
        for act in range(len(leaves)):
            env_cpy = deepcopy(env)
            state, rew, done,  _ = env_cpy.step(act)
            acc_rew = rew
            step = 1
            while not done:
                action = np.random.randint(0, len(leaves))
                _, rew, done, _ = env_cpy.step(action)
                acc_rew = acc_rew + dc**step * rew
                step+=1
            self.visited_q_nodes[act][state].setValue(acc_rew)
            self.visited_q_nodes[act][state].setQValue(acc_rew, act)


    def getSumTerm(self, action):
        sum = 0.0
      
        for key, Node in self.visited_q_nodes[action].items():
            #visits = max(1, Node.getCombinedVisits())
           
            visits = Node.getCombinedVisits()
            sum+= visits * Node.getValue()
    
      
        return sum


    def getRewards(self, action):
        rewards = []
        for key, Node, in self.visited_q_nodes[action].items():
            visits = Node.getCombinedVisits()
            rewards.extend([Node.getValue()] * visits)

        return np.array(rewards)
