from typing import Any, Dict, List, Tuple
import gym
from waymax import config as _config
from waymax import dynamics
from waymax import env as _env
import dataclasses
from stable_baselines3 import PPO
from scenario_extraction import get_scenario
from Config import Config


def train(config: Config):
    if config.dynamics_model == 'InvertibleBicycleModel':
        dynamics_model = dynamics.InvertibleBicycleModel(
                normalize_actions=False,  # This means we feed in all actions as in [-1, 1]
            )
    else:
        print("Dynamics model not found")
        dynamics_model = None

    # Use relative coordinates
    obs_config = dataclasses.replace(
        _config.ObservationConfig(),
        coordinate_frame=_config.CoordinateFrame.OBJECT,
        roadgraph_top_k=config.roadgraph_top_k,
    )

    # Create env config
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=config.max_num_objects,
        observation=obs_config,
        rewards=_config.LinearCombinationRewardConfig(
            config.waymax_reward_weights
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

        scenarios = get_scenario(config)

        gym.register(
        id = config.wrapper_id,
        entry_point = config.entry_point
        )
        if config.model == 'PPO':
            model = PPO('MultiInputPolicy', env, learning_rate=config.learning_rate, batch_size=config.batch_size, n_epochs=config.n_epochs, verbose=1)
        else:
            print("No other model found, still use PPO as default")
            model = PPO('MultiInputPolicy', env, learning_rate=config.learning_rate, batch_size=config.batch_size, n_epochs=config.n_epochs, verbose=1)

        for i in range(config.num_scenario_used):
            if config.data_usage == 'local':
                scenario = scenarios[i]
            else:
                scenario = next(scenarios)
            env = gym.make(config.wrapper_id, env = waymax_base_env, scenario=scenario, config=config, obs_with_agent_id=False)
            env.reset()
            model.set_env(env)
            model.learn(total_timesteps=config.training_timesteps, reset_num_timesteps=False, tb_log_name="PPO")

        model.save(config.model_save_path)

if __name__ == '__main__':
    config = Config()
    train(config)