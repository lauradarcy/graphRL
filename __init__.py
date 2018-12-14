from gym.envs.registration import register

register(
    id='MyEnv-v0',
    entry_point='graphRL.graphRL:graphRL',
    kwargs={'network_size': 10, 'input_nodes': 3}
)