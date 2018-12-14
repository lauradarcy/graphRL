from gym.envs.registration import register

register(
    id='graphRL-v0',
    entry_point='graphRL.envs:graphRL',
    kwargs={'network_size': 10, 'input_nodes': 3}
)