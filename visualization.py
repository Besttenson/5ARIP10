from typing import Any, Dict, List, Tuple
import gym
from waymax import config as _config
from waymax import dynamics
from waymax import env as _env
import dataclasses
from stable_baselines3 import PPO
from scenario_extraction import get_scenario
from Config import Config 
from jax import numpy as jnp
from waymax import agents
from actors import create_customized_actor
import jax
from waymax import visualization
import mediapy

model = PPO.load(Config.model_save_path)
if Config.dynamics_model == 'InvertibleBicycleModel':
    dynamics_model = dynamics.InvertibleBicycleModel(
            normalize_actions=False,  # This means we feed in all actions as in [-1, 1]
        )
else:
    print("Dynamics model not found")
    dynamics_model = dynamics.InvertibleBicycleModel(
            normalize_actions=False,  # This means we feed in all actions as in [-1, 1]
        )

# Use relative coordinates
obs_config = dataclasses.replace(
    _config.ObservationConfig(),
    coordinate_frame=_config.CoordinateFrame.OBJECT,
    roadgraph_top_k=Config.roadgraph_top_k,
)

# Create env config
env_config = dataclasses.replace(
    _config.EnvironmentConfig(),
    max_num_objects=Config.max_num_objects,
    observation=obs_config,
    rewards=_config.LinearCombinationRewardConfig(
        Config.waymax_reward_weights
    ),
    # Controll all valid objects in the scene.
    controlled_object=_config.ObjectType.VALID,
)

# Create waymax environment
if dynamics_model:
    waymax_base_env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )

gym.register(
id = Config.wrapper_id,
entry_point = Config.entry_point
)
scenario_val = get_scenario(Config())
obj_idx = jnp.arange(32)
env = gym.make(Config.wrapper_id, env=waymax_base_env, scenario=scenario_val[0], config=Config(), obs_with_agent_id=False)
dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=False)
states = [waymax_base_env.reset(scenario_val[0])]
index = states[0].object_metadata.is_sdc


for num, idx in enumerate(index):
    if idx:
      ego_index = num
      break
    else:
      ego_index = 0
print("ego_index: ", ego_index)
sim_actors = agents.create_constant_speed_actor(dynamics_model = dynamics_model, is_controlled_func=lambda state: obj_idx != ego_index)
ego_actor = create_customized_actor(dynamics_model = dynamics_model, is_controlled_func=lambda state: obj_idx == ego_index)
# selfactors = create_customized_actor(dynamics_model = dynamics_model, is_controlled_func=lambda state: obj_idx != -1)
# Reset the environment to start the validation episode
obs = env.reset()


for i in range(states[0].remaining_timesteps):
    action, _states = model.predict(obs, deterministic=False)

    # print((obs))
    obs, reward, done, info = env.step(action)

    current_state = states[-1]

    actors = [ego_actor, sim_actors]
    jit_step = jax.jit(waymax_base_env.step)
    jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

    outputs = [jit_select_action(action, current_state, None, None) for jit_select_action in jit_select_action_list]


    action_update = agents.merge_actions(outputs)

    next_state = waymax_base_env.step(current_state, action_update)
    states.append(next_state)

imgs = []
states = jax.device_put(states, jax.devices('cpu')[0])
for state in states:
  imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
mediapy.show_video(imgs, fps=10)