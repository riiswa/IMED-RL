import numpy as np
import random
from sklearn.utils.extmath import cartesian

class CustomSimpleEnv:
    def __init__(self):
        self.currentState = 1
        self.currentStep = 0

    def step(self, action):
        self.currentStep+=1
        if self.currentState == 5 and action == 1:
            self.currentState = 0
        else:
            if action == 0:
                self.currentState = self.currentState - 1
            else:
                self.currentState = self.currentState +1

        done = False
        if self.currentState == 0 or self.currentStep == 50:
            done = True
        if self.currentState == 0:
            return 0, 0.0, done, ""
        elif self.currentState == 1:
            return 1, 0.0, done, ""
        elif self.currentState == 2:
            return 2,0.0, done, ""
        elif self.currentState == 3:
            return 3, 0.0, done, ""
        elif self.currentState == 4:
            return 4, 0.0, done, ""
        else:
            return 5, 1.0, done, ""

    def reset(self):
        self.currentState = 1
        self.currentStep = 0
        return 1

    def close(self):
        pass
    def seed(self, i):
        pass

class CustomComplexEnv:
    def __init__(self):
        self.currentState = 1
        self.currentStep = 0

    def step(self, action):
        self.currentStep+=1
        if (self.currentState == 4 or self.currentState == 5) and action == 1:
            self.currentState = 0



        done = False
        if self.currentStep == 75:
            done = True


        if self.currentState == 1:
            if action == 0:
                self.currentState = 0
            else:
                if random.random() > 0.5:
                    self.currentState = 2
                else:
                    self.currentState = 3
        elif self.currentState == 2:
            if action == 0:
                self.currentState = 1
            else:
               self.currentState = 4
        elif self.currentState == 3:
            if action == 0:
                self.currentState = 1
            else:
                if random.random() > 0.5:
                    self.currentState = 5
                else:
                    self.currentState = 5
        elif self.currentState == 4:
            if action == 0:
                if random.random() > 0.5:
                    self.currentState = 2
                else:
                    self.currentState = 3
            else:
               self.currentState = 0
        elif self.currentState == 5:
            if action == 0:
               self.currentState = 3
            else:
               self.currentState = 0



        if self.currentState == 0:
            done = True
            return np.array([self.currentState, self.currentState]), 0.0, done, ""
        elif self.currentState == 1:
            return np.array([self.currentState, self.currentState]), 0.9, done, ""
        elif self.currentState == 2:
            return np.array([self.currentState, self.currentState]), 1.2, done, ""
        elif self.currentState == 3:
            return np.array([self.currentState, self.currentState]), 1.0, done, ""
        elif self.currentState == 4:
            return np.array([self.currentState, self.currentState]), 1.5, done, ""
        elif self.currentState == 5:
            return np.array([self.currentState, self.currentState]), 0.2, done, ""


    def reset(self):
        self.currentState = 1
        self.currentStep = 0
        return np.array([self.currentState, self.currentState])

    def close(self):
        pass


import gym
from gym import spaces
from gym.utils import seeding

class NChainEnv(gym.Env):
    """n-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, n=5, slip=0.2, small=2.0/10.0, large=10.00/10.0):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if action:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = 0
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = 0
            self.state += 1
        else:  # 'forwards': stay at the end of the chain, collect large reward
            reward = self.large
        done = False
        reward = float(reward) / float(self.large)
        return self.state, reward , done, {}

    def reset(self):
        self.state = 0
        return self.state

import numpy as np
#from mushroom_rl.environments.finite_mdp import FiniteMDP
import sys
import json
import torch
import pickle
import numpy as np

from copy import deepcopy
from pathlib import Path

if sys.version_info >= (3, 7):
    from zipfile import ZipFile
else:
    from zipfile37 import ZipFile


class Serializable(object):
    """
    Interface to implement serialization of a MushroomRL object.
    This provide load and save functionality to save the object in a zip file.
    It is possible to save the state of the agent with different levels of

    """
    def save(self, path, full_save=False):
        """
        Serialize and save the object to the given path on disk.

        Args:
            path (Path, str): Relative or absolute path to the object save
                location;
            full_save (bool): Flag to specify the amount of data to save for
                MushroomRL data structures.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with ZipFile(path, 'w') as zip_file:
            self.save_zip(zip_file, full_save)

    def save_zip(self, zip_file, full_save, folder=''):
        """
        Serialize and save the agent to the given path on disk.

        Args:
            zip_file (ZipFile): ZipFile where te object needs to be saved;
            full_save (bool): flag to specify the amount of data to save for
                MushroomRL data structures;
            folder (string, ''): subfolder to be used by the save method.
        """
        primitive_dictionary = dict()

        for att, method in self._save_attributes.items():

            if not method.endswith('!') or full_save:
                method = method[:-1] if method.endswith('!') else method
                attribute = getattr(self, att) if hasattr(self, att) else None

                if attribute is not None:
                    if method == 'primitive':
                        primitive_dictionary[att] = attribute
                    elif method == 'none':
                        pass
                    elif hasattr(self, '_save_{}'.format(method)):
                        save_method = getattr(self, '_save_{}'.format(method))
                        file_name = "{}.{}".format(att, method)
                        save_method(zip_file, file_name, attribute,
                                    full_save=full_save, folder=folder)
                    else:
                        raise NotImplementedError(
                            "Method _save_{} is not implemented for class '{}'".
                                format(method, self.__class__.__name__)
                        )

        config_data = dict(
            type=type(self),
            save_attributes=self._save_attributes,
            primitive_dictionary=primitive_dictionary
        )

        self._save_pickle(zip_file, 'config', config_data, folder=folder)

    @classmethod
    def load(cls, path):
        """
        Load and deserialize the agent from the given location on disk.

        Args:
            path (Path, string): Relative or absolute path to the agents save
                location.

        Returns:
            The loaded agent.

        """
        path = Path(path)
        if not path.exists():
            raise ValueError("Path to load agent is not valid")

        with ZipFile(path, 'r') as zip_file:
            loaded_object = cls.load_zip(zip_file)

        return loaded_object

    @classmethod
    def load_zip(cls, zip_file, folder=''):
        config_path = Serializable._append_folder(folder, 'config')

        try:
            object_type, save_attributes, primitive_dictionary = \
                cls._load_pickle(zip_file, config_path).values()
        except KeyError:
            return None

        if object_type is list:
            return cls._load_list(zip_file, folder, primitive_dictionary['len'])
        else:
            loaded_object = object_type.__new__(object_type)
            setattr(loaded_object, '_save_attributes', save_attributes)

            for att, method in save_attributes.items():
                mandatory = not method.endswith('!')
                method = method[:-1] if not mandatory else method
                file_name = Serializable._append_folder(
                    folder, '{}.{}'.format(att, method)
                )

                if method == 'primitive' and att in primitive_dictionary:
                    setattr(loaded_object, att, primitive_dictionary[att])
                elif file_name in zip_file.namelist() or \
                        (method == 'mushroom' and mandatory):
                    load_method = getattr(cls, '_load_{}'.format(method))
                    if load_method is None:
                        raise NotImplementedError('Method _load_{} is not'
                                                  'implemented'.format(method))
                    att_val = load_method(zip_file, file_name)
                    setattr(loaded_object, att, att_val)

                else:
                    setattr(loaded_object, att, None)

            loaded_object._post_load()

            return loaded_object

    @classmethod
    def _load_list(self, zip_file, folder, length):
        loaded_list = list()

        for i in range(length):
            element_folder = Serializable._append_folder(folder, str(i))
            loaded_element = Serializable.load_zip(zip_file, element_folder)
            loaded_list.append(loaded_element)

        return loaded_list

    def copy(self):
        """
        Returns:
             A deepcopy of the agent.

        """
        return deepcopy(self)

    def _add_save_attr(self, **attr_dict):
        """
        Add attributes that should be saved for an agent.
        For every attribute, it is necessary to specify the method to be used to
        save and load.
        Available methods are: numpy, mushroom, torch, json, pickle, primitive
        and none. The primitive method can be used to store primitive attributes,
        while the none method always skip the attribute, but ensure that it is
        initialized to None after the load. The mushroom method can be used with
        classes that implement the Serializable interface. All the other methods
        use the library named.
        If a "!" character is added at the end of the method, the field will be
        saved only if full_save is set to True.

        Args:
            **attr_dict: dictionary of attributes mapped to the method
                that should be used to save and load them.

        """
        if not hasattr(self, '_save_attributes'):
            self._save_attributes = dict()
        self._save_attributes.update(attr_dict)

    def _post_load(self):
        """
        This method can be overwritten to implement logic that is executed
        after the loading of the agent.

        """
        pass

    @staticmethod
    def _append_folder(folder, name):
        if folder:
           return folder + '/' + name
        else:
           return name

    @staticmethod
    def _load_pickle(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return pickle.load(f)

    @staticmethod
    def _load_numpy(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return np.load(f)

    @staticmethod
    def _load_torch(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return torch.load(f)

    @staticmethod
    def _load_json(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_mushroom(zip_file, name):
        return Serializable.load_zip(zip_file, name)

    @staticmethod
    def _save_pickle(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    @staticmethod
    def _save_numpy(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            np.save(f, obj)

    @staticmethod
    def _save_torch(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            torch.save(obj, f)

    @staticmethod
    def _save_json(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            string = json.dumps(obj)
            f.write(string.encode('utf8'))

    @staticmethod
    def _save_mushroom(zip_file, name, obj, folder, full_save):
        new_folder = Serializable._append_folder(folder, name)
        if isinstance(obj, list):
            config_data = dict(
                type=list,
                save_attributes=dict(),
                primitive_dictionary=dict(len=len(obj))
            )

            Serializable._save_pickle(zip_file, 'config', config_data, folder=new_folder)
            for i, element in enumerate(obj):
                element_folder = Serializable._append_folder(new_folder, str(i))
                element.save_zip(zip_file, full_save=full_save, folder=element_folder)
        else:
            obj.save_zip(zip_file, full_save=full_save, folder=new_folder)

    @staticmethod
    def _get_serialization_method(class_name):
        if issubclass(class_name, Serializable):
            return 'mushroom'
        else:
            return 'pickle'
class MDPInfo(Serializable):
    """
    This class is used to store the information of the environment.

    """
    def __init__(self, observation_space, action_space, gamma, horizon):
        """
        Constructor.

        Args:
             observation_space ([Box, Discrete]): the state space;
             action_space ([Box, Discrete]): the action space;
             gamma (float): the discount factor;
             horizon (int): the horizon.

        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon

        self._add_save_attr(
            observation_space='mushroom',
            action_space='mushroom',
            gamma='primitive',
            horizon='primitive'
        )

    @property
    def size(self):
        """
        Returns:
            The sum of the number of discrete states and discrete actions. Only
            works for discrete spaces.

        """
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        """
        Returns:
            The concatenation of the shape tuple of the state and action
            spaces.

        """
        return self.observation_space.shape + self.action_space.shape
class Environment(object):
    """
    Basic interface used by any mushroom environment.

    """

    @classmethod
    def register(cls):
        """
        Register an environment in the environment list.

        """
        env_name = cls.__name__

        if env_name not in Environment._registered_envs:
            Environment._registered_envs[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered environments.

        Returns:
             The list of the registered environments.

        """
        return list(Environment._registered_envs.keys())

    @staticmethod
    def make(env_name, *args, **kwargs):
        """
        Generate an environment given an environment name and parameters.
        The environment is created using the generate method, if available. Otherwise, the constructor is used.
        The generate method has a simpler interface than the constructor, making it easier to generate
        a standard version of the environment. If the environment name contains a '.' separator, the string
        is splitted, the first element is used to select the environment and the other elements are passed as
        positional parameters.

        Args:
            env_name (str): Name of the environment,
            *args: positional arguments to be provided to the environment generator;
            **kwargs: keyword arguments to be provided to the environment generator.

        Returns:
            An instance of the constructed environment.

        """

        if '.' in env_name:
            env_data = env_name.split('.')
            env_name = env_data[0]
            args = env_data[1:] + list(args)

        env = Environment._registered_envs[env_name]

        if hasattr(env, 'generate'):
            return env.generate(*args, **kwargs)
        else:
            return env(*args, **kwargs)

    def __init__(self, mdp_info):
        """
        Constructor.

        Args:
             mdp_info (MDPInfo): an object containing the info of the
                environment.

        """
        self._mdp_info = mdp_info

    def seed(self, seed):
        """
        Set the seed of the environment.

        Args:
            seed (float): the value of the seed.

        """
        if hasattr(self, 'env') and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        else:
            pass

    def reset(self, state=None):
        """
        Reset the current state.

        Args:
            state (np.ndarray, None): the state to set to the current state.

        Returns:
            The current state.

        """
        raise NotImplementedError

    def step(self, action):
        """
        Move the agent from its current state according to the action.

        Args:
            action (np.ndarray): the action to execute.

        Returns:
            The state reached by the agent executing ``action`` in its current
            state, the reward obtained in the transition and a flag to signal
            if the next state is absorbing. Also an additional dictionary is
            returned (possibly empty).

        """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def stop(self):
        """
        Method used to stop an mdp. Useful when dealing with real world
        environments, simulators, or when using openai-gym rendering

        """
        pass

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info

    @staticmethod
    def _bound(x, min_value, max_value):
        """
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;

        Returns:
            The bounded variable.

        """
        return np.maximum(min_value, np.minimum(x, max_value))

    _registered_envs = dict()
class FiniteMDP(Environment):
    """
    Finite Markov Decision Process.

    """
    def __init__(self, p, rew, mu=None, gamma=.9, horizon=np.inf):
        """
        Constructor.

        Args:
            p (np.ndarray): transition probability matrix;
            rew (np.ndarray): reward matrix;
            mu (np.ndarray, None): initial state probability distribution;
            gamma (float, .9): discount factor;
            horizon (int, np.inf): the horizon.

        """
        assert p.shape == rew.shape
        assert mu is None or p.shape[0] == mu.size

        # MDP parameters
        self.p = p
        self.r = rew
        self.mu = mu

        # MDP properties
        observation_space = spaces.Discrete(p.shape[0])
        action_space = spaces.Discrete(p.shape[1])
        horizon = horizon
        gamma = gamma
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self._state = np.array(
                    [np.random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array([np.random.choice(self.p.shape[0])])
        else:
            self._state = state

        return self._state[0]

    def step(self, action):
        p = self.p[self._state[0], action, :]
        next_state = np.array([np.random.choice(p.size, p=p)])
        absorbing = not np.any(self.p[next_state[0]])
        reward = self.r[self._state[0], action, next_state[0]]

        self._state = next_state

        return self._state[0], reward, absorbing, {}


def generate_river(n=6, gamma=0.95, small=0.0005, large=1, horizon=np.inf):
    nA = 2
    nS = n
    p = compute_probabilities_river(nS, nA)
    r = compute_rewards_river(nS, nA, small, large)
    mu = compute_mu_river(nS)
    return FiniteMDP(p, r, mu, gamma, horizon)


def compute_probabilities_river(nS, nA):
    p = np.zeros((nS, nA, nS))
    for i in range(1, nS):
        p[i, 0, i - 1] = 1
        if i != nS - 1:
            p[i, 1, i - 1] = 0.1
            p[i, 1, i] = 0.6
        else:
            p[i, 1, i - 1] = 0.7
            p[i, 1, i] = 0.3
    for i in range(nS - 1):
        p[i, 1, i + 1] = 0.3
    # state 0
    p[0, 0, 0] = 1
    p[0, 1, 0] = 0.7

    return p


def compute_rewards_river(nS, nA, small, large):
    r = np.zeros((nS, nA, nS))
    r[0, 0, 0] = float(small)/float(large)
    r[nS - 1, 1, nS - 1] = float(large)/float(large)
    return r


def compute_mu_river(nS):
    mu = np.zeros(nS)
    mu[1] = 0.5
    mu[2] = 0.5
    return mu

def generate_arms(gamma=0.95, horizon=np.inf):
    nA = 6
    nS = 7
    rew = [50.0/6000.0, 133.0/6000.0, 300.0/6000.0, 800.0/6000.0, 1660.0/6000.0, 6000.0/6000.0]
    p = compute_probabilities_arms(nS, nA)
    r = compute_rewards_arms(nS, nA, rew)
    mu = compute_mu_arms(nS)
    return FiniteMDP(p, r, mu, gamma, horizon)


def compute_probabilities_arms(nS, nA):
    p = np.zeros((nS, nA, nS))
    # state 1
    p[0, 0, 1] = 1
    p[1, 0, 1] = p[1, 1, 1] = p[1, 2, 1] = p[1, 3, 1] = p[1, 5, 1] = 1
    p[1, 4, 0] = 1
    # state 2
    p[0, 1, 2] = 0.15
    p[0, 1, 0] = 0.85
    p[2, 0, 0] = p[2, 2, 0] = p[2, 3, 0] = p[2, 4, 0] = p[2, 5, 0] = 1
    p[2, 1, 2] = 1
    # state 3
    p[0, 2, 3] = 0.1
    p[0, 2, 0] = 0.9
    p[3, 2, 3] = 1
    p[3, 0, 0] = p[3, 1, 0] = p[3, 3, 0] = p[3, 4, 0] = p[3, 5, 0] = 1
    # state 4
    p[0, 3, 4] = 0.05
    p[0, 3, 0] = 0.95
    p[4, 3, 4] = 1
    p[4, 0, 0] = p[4, 1, 0] = p[4, 2, 0] = p[4, 4, 0] = p[4, 5, 0] = 1
    # state 5
    p[0, 4, 5] = 0.03
    p[0, 4, 0] = 0.97
    p[5, 4, 5] = 1
    p[5, 0, 0] = p[5, 1, 0] = p[5, 2, 0] = p[5, 3, 0] = p[5, 5, 0] = 1
    # state 6
    p[0, 5, 6] = 0.01
    p[0, 5, 0] = 0.99
    p[6, 5, 6] = 1
    p[6, 0, 0] = p[6, 1, 0] = p[6, 2, 0] = p[6, 3, 0] = p[6, 4, 0] = 1
    return p


def compute_rewards_arms(nS, nA, rew):
    r = np.zeros((nS, nA, nS))
    r[1, 0, 1] = r[1, 1, 1] = r[1, 2, 1] = r[1, 3, 1] = r[1, 5, 1] = rew[0]
    for i in range(2, nS):
        r[i, i - 1, i] = rew[i - 1]
    return r


def compute_mu_arms(nS):
    mu = np.zeros(nS)
    mu[0] = 1
    return mu


class RandomGrid:
    def __init__(self):
        self.state = 0
        self.x = 0
        self. y = 0
        self.p_layers = [0.25, 0.15, 0.1]
        self.max_x = 3
        self.max_y = 3
        self.rewards = np.ones((self.max_x + 1, self.max_y +1)) * 0.15
        self.rewards[3, 0] = 0.7
        self.rewards[1, 1] = 0.6
        self.rewards[2, 3] = 0.9
        self.rewards[3, 2] = 0.9
        self.rewards[3,3] = 1.0
    def step(self, action):
        #0 right
        #2 down
        #3 collect reward
        #4 left
        #5 up
        reward = 0.0
        done = False
        if action == 0:
            if self.x != self.max_x:
                if np.random.random() < self.p_layers[self.x]:
                    self.x = self.x + 1
        elif action == 1:
            if self.y != self.max_y:
                if np.random.random() < self.p_layers[self.y]:
                    self.y = self.y + 1
        elif action == 2:
            reward = self.rewards[self.x, self.y]
            if reward > 0:
                done = True
        elif action == 3:
            if self.x != 0:
                self.x = self.x - 1
        elif action == 4:
            if self.y != 0:
                self.y = self.y - 1

        self.state = self.y * (self.max_x +1) + self.x

        return self.state, reward, done, "info"




    def seed(self, seed):
        np.random.seed(seed)
    def reset(self):
        self.state = 0
        self.x = 0
        self.y = 0
        return self.state

    def close(self):
        pass

def generate_taxi(grid, prob=.9, rew=(0, 1.0/15.0, 3.0/15.0, 15.0/15.0), gamma=.99, horizon=np.inf):
    """
    This Taxi generator requires a .txt file to specify the shape of the grid
    world and the cells. There are five types of cells: 'S' is the starting
    where the agent is; 'G' is the goal state; '.' is a normal cell; 'F' is a
    passenger, when the agent steps on a hole, it picks up it.
    '#' is a wall, when the agent is supposed to step on a wall, it actually
    remains in its current state. The initial states distribution is uniform
    among all the initial states provided. The episode terminates when the agent
    reaches the goal state. The reward is always 0, except for the goal state
    where it depends on the number of collected passengers. Each action has
    a certain probability of success and, if it fails, the agent goes in a
    perpendicular direction from the supposed one.

    The grid is expected to be rectangular.

    This problem is inspired from:
    "Bayesian Q-Learning". Dearden R. et al.. 1998.

    Args:
        grid (str): the path of the file containing the grid structure;
        prob (float, .9): probability of success of an action;
        rew (tuple, (0, 1, 3, 15)): rewards obtained in goal states;
        gamma (float, .99): discount factor;
        horizon (int, np.inf): the horizon.

    Returns:
        A FiniteMDP object built with the provided parameters.

    """
    grid_map, cell_list, passenger_list = parse_grid_taxi(grid)

    assert len(rew) == len(np.argwhere(np.array(grid_map) == 'F')) + 1

    p = compute_probabilities_taxi(grid_map, cell_list, passenger_list, prob)
    r = compute_reward_taxi(grid_map, cell_list, passenger_list, rew)
    mu = compute_mu_taxi(grid_map, cell_list, passenger_list)

    return FiniteMDP(p, r, mu, gamma, horizon)


def parse_grid_taxi(grid):
    """
    Parse the grid file:

    Args:
        grid (str): the path of the file containing the grid structure.

    Returns:
        A list containing the grid structure.

    """
    grid_map = list()
    cell_list = list()
    passenger_list = list()
    with open(grid, 'r') as f:
        m = f.read()

        assert 'S' in m and 'G' in m

        row = list()
        row_idx = 0
        col_idx = 0
        for c in m:
            if c in ['#', '.', 'S', 'G', 'F']:
                row.append(c)
                if c in ['.', 'S', 'G', 'F']:
                    cell_list.append([row_idx, col_idx])
                    if c == 'F':
                        passenger_list.append([row_idx, col_idx])
                col_idx += 1
            elif c == '\n':
                grid_map.append(row)
                row = list()
                row_idx += 1
                col_idx = 0
            else:
                raise ValueError('Unknown marker.')

    return grid_map, cell_list, passenger_list


def compute_probabilities_taxi(grid_map, cell_list, passenger_list, prob):
    """
    Compute the transition probability matrix.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        passenger_list (list): list of passenger cells;
        prob (float): probability of success of an action.

    Returns:
        The transition probability matrix;

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list) * 2**len(passenger_list)
    p = np.zeros((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    passenger_states = cartesian([[0, 1]] * len(passenger_list))

    for i in range(n_states):
        idx = i // len(cell_list)
        collected_passengers = np.array(
            passenger_list)[np.argwhere(passenger_states[idx] == 1).ravel()]
        state = c[i % len(cell_list)]

        if g[tuple(state)] in ['.', 'S', 'F']:
            if g[tuple(state)] in ['F']\
                    and state.tolist() not in collected_passengers.tolist():
                continue
            for a in range(len(directions)):
                new_state = state + directions[a]

                j = np.where((c == new_state).all(axis=1))[0]
                if j.size > 0:
                    assert j.size == 1

                    if g[tuple(new_state)] == 'F' and new_state.tolist()\
                            not in collected_passengers.tolist():
                        current_passenger_state = np.zeros(len(passenger_list))
                        current_passenger_idx = np.where(
                            (new_state == passenger_list).all(axis=1))[0]
                        current_passenger_state[current_passenger_idx] = 1
                        new_passenger_state = passenger_states[
                            idx] + current_passenger_state
                        new_idx = np.where((
                            passenger_states == new_passenger_state).all(
                            axis=1))[0]

                        j += len(cell_list) * new_idx
                    else:
                        j += len(cell_list) * idx
                else:
                    j = i

                p[i, a, j] = prob

                for d in [1 - np.abs(directions[a]),
                          np.abs(directions[a]) - 1]:
                    slip_state = state + d
                    k = np.where((c == slip_state).all(axis=1))[0]
                    if k.size > 0:
                        assert k.size == 1

                        if g[tuple(slip_state)] == 'F' and slip_state.tolist()\
                                not in collected_passengers.tolist():
                            current_passenger_state = np.zeros(
                                len(passenger_list))
                            current_passenger_idx = np.where(
                                (slip_state == passenger_list).all(axis=1))[0]
                            current_passenger_state[current_passenger_idx] = 1
                            new_passenger_state = passenger_states[
                                idx] + current_passenger_state
                            new_idx = np.where((
                                passenger_states == new_passenger_state).all(
                                axis=1))[0]

                            k += len(cell_list) * new_idx
                        else:
                            k += len(cell_list) * idx
                    else:
                        k = i

                    p[i, a, k] += (1. - prob) * .5

    return p


def compute_reward_taxi(grid_map, cell_list, passenger_list, rew):
    """
    Compute the reward matrix.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        passenger_list (list): list of passenger cells;
        rew (tuple): rewards obtained in goal states.

    Returns:
        The reward matrix.

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list) * 2**len(passenger_list)
    r = np.zeros((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    passenger_states = cartesian([[0, 1]] * len(passenger_list))

    for goal in np.argwhere(g == 'G'):
        for a in range(len(directions)):
            prev_state = goal - directions[a]
            if prev_state.tolist() in c.tolist():
                for i in range(len(passenger_states)):
                    i_idx = np.where((c == prev_state).all(axis=1))[0] + len(
                        cell_list) * i
                    j_idx = j = np.where((c == goal).all(axis=1))[0] + len(
                        cell_list) * i

                    r[i_idx, a, j_idx] = rew[np.sum(passenger_states[i])]

    return r


def compute_mu_taxi(grid_map, cell_list, passenger_list):
    """
    Compute the initial states distribution.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        passenger_list (list): list of passenger cells.

    Returns:
        The initial states distribution.

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list) * 2**len(passenger_list)
    mu = np.zeros(n_states)
    starts = np.argwhere(g == 'S')

    for s in starts:
        i = np.where((c == s).all(axis=1))[0]
        mu[i] = 1. / len(starts)

    return mu