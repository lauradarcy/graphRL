import gym
from graph_tool import Graph
from gym import spaces
from gym.utils import seeding
import numpy as np
from graph_tool.all import *


class graphRL(gym.Env):
    """
    will have fixed action space, but not all actions are valid within each state
    step function should have a function that tests if the chosen action is valid
    the observation returned will be the graph_tool graph, but the state will just be
    the adjacency matrix (? maybe, currently have obs space as the matrix)
    maybe step function just alters the given graph
    """
    metadata = {'render.modes': ['human', 'graph']}

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
        self.time_step = 0
        self.observation = adjacency(self.graph).toarray()

        self.reset()

    def render(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.graph
        elif mode == 'human':
            filename = "./renders/render" + str(self.time_step) + ".png"
            graph_draw(self.graph, vertex_text=self.graph.vertex_index, vertex_font_size=18,
                       output_size=(1000, 1000), output=filename)
            pass

    def step(self, action):
        assert self.action_space.contains(action)
        valid_source_nodes = [index for index, in_degree in
                              enumerate(self.graph.get_in_degrees(self.graph.get_vertices())) if
                              (in_degree > 0 or index < self.input_nodes)]
        if action[0] not in valid_source_nodes:
            raise ValueError('this action does not have a valid from node')
        new_edge = self.graph.add_edge(action[0], action[1])
        if not is_DAG(self.graph):
            self.graph.remove_edge(new_edge)
            raise ValueError('this action violates the DAG property')
        self.observation = adjacency(self.graph).toarray()
        if not self.observation_space.contains(self.observation):
            self.graph.remove_edge(new_edge)
            self.observation = adjacency(self.graph).toarray()
            raise ValueError('this action makes a duplicate edge')

        return self.observation, reward, done, {"time_step": self.time_step}

    def reset(self):
        self.graph.clear_edges()
        self.time_step = 0
        self.observation = adjacency(self.graph).toarray()
        return self.observation
