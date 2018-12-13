# graph tool reinforcement learning

practice making an openAI gym environment for a tabular graph solver

basic code for environment:

```python
class Env(object):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        logger.warn("Could not seed environment %s", self)
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)
```

## `__init__`:

```python
    def __init__(self, network_size=10, input_nodes=3):
        self.network_size = network_size
        self.input_nodes = input_nodes
        self.graph = Graph()
        self.graph.set_fast_edge_removal(True)
        self.graph.add_vertex(self.network_size)

        self.action_space = spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))
        self.observation_space = spaces.MultiDiscrete(np.full((self.network_size, self.network_size), 2))
        
        self.time_step = 0
        self.observation = adjacency(self.graph).toarray()
        
        self.reset()
```

#### `network_size` and `input_nodes`:

Network size is the total number of services, input nodes are where the "source data" is coming from - all further connections either have to come from an input node, or from a node that has previously received information that came from an input node (recursively).

#### `action_space`:

 should be the list of possible vertex to vertex edges. Note that this should NOT account for the DAG compatibility - we want the action space to be the same for all states, so instead should simply be a tuple that contains all node indices in the graph, i.e. a possible action could be `(0,3)` - make a directed connection from node 0 to node 3. the entire action space should then be `spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))` - a tuple that allows for an integer value from 0 up to the final node index.

#### `observation_space`:

should allow for possible valid state spaces. The state space for this tabular function will simply be the adjacency matrix of the network - and we want a network with only one edge going from one node to another, so we can make the observation space a matrix the size of the adjacency matrix (a square matrix with length of the number of nodes), filled with the value 2. This means the observation space will only be valid for matrices where all values are either the integers 0 or 1.

## `step(self, action)`:

```python
    def step(self, action):
        assert self.action_space.contains(action)
        valid_source_nodes = [index for index, in_degree in                         							enumerate(self.graph.get_in_degrees(self.graph.get_vertices()))
                        if (in_degree > 0 or index < self.input_nodes)]
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
```

#### `raise ValueError()`:

these are used to ensure the action given is actually a valid action. If they aren't, the method breaks. When using this environment, you will need to wrap the `step` method in a `try` clause.





