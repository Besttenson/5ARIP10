from jax import numpy as jnp
from typing import Dict, List, Tuple

class Config:
    # Scenarios config
    data_usage: str = 'local'                    # Use local or remote scenario
    local_catagorized_data_path: str = "SCLaneChange_new"
    local_scenario_data_path: str = 'tf_dataset/training_tfexample.tfrecord@3'
    num_scenario_used: int = 1                   # Number of scenarios used in training

    # Waymax env config
    max_num_objects: int = 32
    waymax_reward_weights: Dict[str, float] = {'offroad': -1.0, 'overlap': -1.0}
    roadgraph_top_k: int = 500
    dynamics_model: str = 'InvertibleBicycleModel' 

    # Gym config
    model_save_path: str = 'model/ppo'           
    wrapper_id: str = 'gym_wrapper/WaymaxWrapper'
    entry_point: str = 'gym_env:WaymaxWrapper'
    mode: str = 'simple'                         # Use simple obs and reward to get a quick result or not
    action_low: List[float] = [-10, -(jnp.pi)]   # Action space low bound
    action_high: List[float] = [10, jnp.pi]      # Action space high bound
    safety_check: bool = False                   # Whether to use safety check in lane change scenario

    # Training config
    training_timesteps: int = 8000
    model: str = 'PPO'                           # Model used in training
    learning_rate: float = 0.0003                # Learning rate
    batch_size: int = 32                         # Batch size
    n_epochs: int = 10                           # Number of epochs