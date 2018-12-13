from gym.envs.registration import register

register(
    id='MyEnv-v0',
    entry_point='graphRL.graph_env:raph_tool_env',
)