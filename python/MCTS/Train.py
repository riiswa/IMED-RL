import collections
import multiprocessing
import random
import warnings
from datetime import datetime

import numpy as np
import time
from WMCTS import WassersteinTree
from copy import deepcopy
from CustomEnv import CustomSimpleEnv, CustomComplexEnv, NChainEnv, RandomGrid
import CustomEnv
from PowerUCTMCTS import PUCTMCTS_Tree, IMEDMCTS_Tree
import gym
from tqdm.auto import tqdm



def play_game(env, mcts, mcts_searches, mcts_batch_size, amount_of_actions, maxStep, env_name):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    state = env.reset()

    discr_state, state_idx = mcts.getStateIndex(state)
    mcts.setStartV(state_idx, amount_of_actions)

    prev_state = discr_state
    step = 0


    result = 0

    is_done = False
    while not is_done:

        seed = mcts.search_batch(deepcopy(env), mcts_searches, mcts_batch_size, state, amount_of_actions, step)
        _, idx = mcts.getStateIndex(state)
        action, visit_act = mcts.getActionPolicy(idx)

        state, r, is_done, _ = env.step(action)
        discr_state, state_idx = mcts.getStateIndex(state)
        mcts.updateStartV(state_idx, action, amount_of_actions, env, is_done)
        result += 0.99 ** step * r

        prev_state = discr_state
        if is_done or step > maxStep:
            break

        step += 1


    return result, step, seed


def train(file_name, seed_start, seeds_,  MCTS_SEARCHES, P, initSTD, C, mcts_, mcts_data, maxStep, amount_actions, env_name, usePretrained, pool, pbar):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    seeds = []

    start = seed_start
    for k in range(seeds_):
        seed = start + k
        seeds.append(seed)


    for s in range(len(seeds)):
        random.seed(seeds[s])
        np.random.seed(seeds[s])



        #env = generate_taxi('grid.txt')
        if env_name == "FrozenLake-v1":
           env = gym.make(env_name,  map_name="4x4", is_slippery= True)


        #env = NChainEnv()
        #env.seed(seeds[s])
        #env = CustomSimpleEnv()
        elif env_name == "River":
            env = CustomEnv.generate_river()
        elif env_name == "Arms":
            env = CustomEnv.generate_arms()
        elif env_name == "NChain-v1":
            env = NChainEnv()
        elif env_name == "RandomGrid":
            env = RandomGrid()
        elif env_name == "Taxi":
            env = CustomEnv.generate_taxi('grid.txt')
        env.seed(seeds[s])
       


        if mcts_=="PUCT":
            mcts = PUCTMCTS_Tree(env_name, "default", amount_actions, initSTD,C, P,  mcts_data,  maxStep, usePretrained)
        elif mcts_ == "optWstein":
            mcts = WassersteinTree(env_name, "OptimisticTree",  amount_actions, initSTD,C, P, mcts_data,  maxStep, usePretrained)
        elif mcts_ == "IMED":
            mcts = IMEDMCTS_Tree(env_name, "default", amount_actions, initSTD,C, P,  mcts_data,  maxStep, usePretrained)
        else:
            mcts = WassersteinTree(env_name, "ThompsonTree", amount_actions, initSTD,C, P, mcts_data,  maxStep, usePretrained)

        folder = 'results/'

        def callback(result):
            rew, steps, current_seed = result
            file = open(folder + file_name, 'a')
            file.write(str(steps))
            file.write(" ")
            file.write(str(rew))
            file.write("\n")
            file.close()
            #print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - Done {mcts_}, rollout={MCTS_SEARCHES}, results=({result})")
            pbar.update(1)

        pool.apply_async(play_game, (env, mcts, MCTS_SEARCHES, 1, amount_actions, maxStep, env_name), callback=callback)


if __name__ == "__main__":
    processes = []

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 4)
    i = 0
    progress_bars = []
    for sim in np.logspace(2, 5, 10).astype(int).tolist():
        env_name = "FrozenLake-v1" #FrozenLake-v1 NChain-v1 River Arms
        amount_actions = 4
        init_Std = 0.5
        C = 1.4142
        usePretrained = False
        pretrained_data = 'f_lake.txt'
        maxSteps = 100
        outputfile = f"puct-output-{str(sim).zfill(6)}.txt"
        outputfile2 = f"imed-output-{str(sim).zfill(6)}.txt"
        startseed = 20
        seeds = 32
        simulations = sim
        p = 1.5
        alg = 'PUCT' #wstein, optWstein or PUCT
        alg2 = 'IMED'

        pbar = tqdm(desc=f"PUCT-{sim}".ljust(12), position=i, total=seeds, leave=True)
        progress_bars.append(pbar)
        train(outputfile, startseed, seeds, simulations, p, init_Std, C, alg, deepcopy(pretrained_data), maxSteps, amount_actions, env_name, usePretrained, pool, pbar)
        i += 1
        pbar = tqdm(desc=f"IMED-{sim}".ljust(12), position=i, total=seeds, leave=True)
        progress_bars.append(pbar)
        train(outputfile2, startseed, seeds, simulations, p, init_Std, C, alg2, deepcopy(pretrained_data), maxSteps,amount_actions, env_name, usePretrained, pool, pbar)
        i += 1

    pool.close()
    pool.join()
    for pbar in progress_bars:
        pbar.close()

