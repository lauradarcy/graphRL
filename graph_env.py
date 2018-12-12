import gym
from graph_tool import Graph
from gym import spaces
from gym.utils import seeding
import numpy as np
from graph_tool.all import *


class graph_tool_env(gym.Env):
    """
    will have fixed action space, but not all actions are valid within each state
    step function should have a function that tests if the chosen action is valid
    the observation returned will be the graph_tool graph, but the state will just be
    the adjacency matrix (? maybe, currently have obs space as the matrix)
    maybe step function just alters the given graph
    """
    graph: Graph

    def __init__(self, network_size=10, input_nodes=3):
        self.network_size = network_size
        self.input_nodes = input_nodes
        self.graph = Graph()
        self.graph.set_fast_edge_removal(True)
        self.graph.add_vertex(self.network_size)

        self.action_space = spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))
        self.observation_space = spaces.MultiDiscrete(np.full((self.network_size, self.network_size), 2))
        # a square matrix of network size, full of the number 2, the multidiscrete space
        # will allow for numbers 0-1 (all positive less than 2)
        self.timestep = 0
        self.number = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

        self.seed()
        self.reset()

    def render(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.graph
        elif mode == 'human':
            filename = "./renders/render"+str(self.timestep)+".png"
            graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=18, output_size=(1000, 1000), output=filename)
            pass

    def step(self, action):
        assert self.action_space.contains(action)

        return self.observation, reward, done, {"number": self.number, "guesses": self.guess_count}

    def reset(self):
        self.number = self.np_random.uniform(-self.range, self.range)
        self.guess_count = 0
        self.observation = 0
        return self.observation
