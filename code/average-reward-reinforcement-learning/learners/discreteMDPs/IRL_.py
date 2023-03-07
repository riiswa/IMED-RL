from learners.discreteMDPs.IRL import IRL

import torch
import torch.nn as nn
import numpy as np


class KInfBlock(nn.Module):
    def __init__(self, nbr_states, nbr_actions):
        super(KInfBlock, self).__init__()
        self.weights = nn.Parameter(torch.zeros((nbr_states, nbr_actions)))

    def forward(self, s, a, p, upper_bound, delta):
        return -torch.sum(p * torch.log(upper_bound - delta * self.weights[s, a]))


class GDIRL(IRL):
    def __init__(self, nbr_states, nbr_actions, name="GRADIENT-DESCENT-IMED-RL",
                 max_iter=3000, epsilon=1e-3, max_reward=1):
        super().__init__(nbr_states, nbr_actions, name, max_iter, epsilon, max_reward)
        self.k_inf_block = KInfBlock(nbr_states, nbr_actions)

    def multinomial_imed(self, state):
        upper_bound = self.max_reward + np.max(self.phi)
        q = self.rewards[state] + self.transitions[state] @ self.phi
        mu = np.max(q)

        for a in range(self.nA):
            if q[a] >= mu:
                self.index[a] = np.log(self.state_action_pulls[state, a])
            else:
                r_d = self.rewards_distributions[state][a]
                vr = np.fromiter(r_d.keys(), dtype=float)
                pr = np.fromiter(r_d.values(), dtype=float)
                pr = pr / pr.sum()

                pt = self.transitions[state][a]

                p = np.zeros(len(pr)*self.nS)
                v = np.zeros(len(pr)*self.nS)
                k = 0
                for i in range(self.nS):
                    for j in range(len(pr)):
                        p[k] = pt[i]*pr[j]
                        v[k] = self.phi[i] + vr[j]
                        k += 1

                delta = v - mu

                x = -self.h(state, a, p, upper_bound, delta)
                n = self.state_action_pulls[state, a]
                self.index[a] = n * x + np.log(x)

    def h(self, s, a, p, upper_bound, delta):
        optimizer = torch.optim.Adam(self.k_inf_block.parameters(), lr=0.01)
        for i in range(100):
            k_inf = self.k_inf_block(s, a, torch.from_numpy(p), upper_bound, torch.from_numpy(delta))
            optimizer.zero_grad()

            k_inf.backward()
            if self.k_inf_block.weights.grad[s, a] < self.epsilon:
                break
            optimizer.step()
        return k_inf.item()
