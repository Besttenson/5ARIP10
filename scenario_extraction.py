import dataclasses
from waymax.config import DatasetConfig, DataFormat
from waymax import dataloader
import json
import os
import time
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from waymax import datatypes
from Config import Config
from waymax import config as _config

def get_scenario(config: Config):
    if config.data_usage == 'local':
        data_iter = decode_bytes(dataloader.get_data_generator(data_config, _preprocess, _postprocess))
        tagged_file_names, host_actors = categorized_scenarios()
        scenarios, _ = extra_filtering(tagged_file_names, host_actors, data_iter)
    else:
        config = dataclasses.replace(_config.WOD_1_1_0_TRAINING, max_num_objects=config.max_num_objects)
        scenarios = dataloader.simulator_state_generator(config=config)

    return scenarios


DATA_LOCAL_01 = DatasetConfig(
    path=Config.local_scenario_data_path,
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
)

data_config = dataclasses.replace(DATA_LOCAL_01, max_num_objects=Config.max_num_objects)

# Write a custom dataloader that loads scenario IDs.
def _preprocess(serialized: bytes) -> dict[str, tf.Tensor]:
    womd_features = dataloader.womd_utils.get_features_description(
        include_sdc_paths=data_config.include_sdc_paths,
        max_num_rg_points=data_config.max_num_rg_points,
        num_paths=data_config.num_paths,
        num_points_per_path=data_config.num_points_per_path,
    )
    womd_features['scenario/id'] = tf.io.FixedLenFeature([1], tf.string)

    deserialized = tf.io.parse_example(serialized, womd_features)
    parsed_id = deserialized.pop('scenario/id')
    deserialized['scenario/id'] = tf.io.decode_raw(parsed_id, tf.uint8)

    return dataloader.preprocess_womd_example(
        deserialized,
        aggregate_timesteps=data_config.aggregate_timesteps,
        max_num_objects=data_config.max_num_objects,
    )

def _postprocess(example: dict[str, tf.Tensor]):
    scenario = dataloader.simulator_state_from_womd_dict(example)
    scenario_id = example['scenario/id']
    return scenario_id, scenario

def decode_bytes(data_iter):
    for scenario_id, scenario in data_iter:
        scenario_id = scenario_id.tobytes().decode('utf-8')
        yield scenario_id, scenario

def categorized_scenarios():
    folder_path = Config.local_catagorized_data_path
    tagged_file_names = []
    host_actors = []

    # Iterate over each file in the json folder
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            with open(Config.local_catagorized_data_path + "/" + file_name, 'r', encoding='utf-8') as file:
                file_content = json.loads(file.read())
                host_actor_numbers = [value["host_actor"] for value in file_content.values()]
                scenario_id = file_name.split("_")
                tagged_file_names.append(scenario_id[2])
                host_actors.append((scenario_id[2], host_actor_numbers[0]))
    return tagged_file_names, host_actors

def overall_angle(yaw):
    angle_change = yaw[0] - yaw[-1]
    return(angle_change)

def avg_velocity(x_velocities, y_velocities):
    velocity_magnitude = 0
    for i in range(len(x_velocities)):
        velocity_magnitude += np.sqrt(x_velocities[i]**2 + y_velocities[i]**2)
    avg_velocity_magnitude = velocity_magnitude/len(x_velocities)
    return avg_velocity_magnitude

def change_ego(states: datatypes.SimulatorState, desired_ego_id:int) -> datatypes.SimulatorState:
    """
    Change the ego vehicle to the id specified in desired_ego_id

    Parameters
    ----------
    states: datatypes.SimulatorState : the current scenario
    desired_ego_id: int: the desired id for the new ego vehicle

    Returns
    -------
    states: datatypes.SimulatorState : the same scenario with the new ego vehicle ID
    """

    index_ego = jnp.argwhere(states.object_metadata.is_sdc)
    if len(index_ego) != 1:
        print('More than 1 ego vehicle in current dataset')
        return states

    index_ego = int(index_ego[0][0])

    states.object_metadata.is_sdc = states.object_metadata.is_sdc.at[index_ego].set(False)
    states.object_metadata.is_sdc = states.object_metadata.is_sdc.at[desired_ego_id].set(True)
    return states

def extra_filtering(tagged_file_names, host_actors, data_iter):
    LaneChangeData = []
    LaneChangeID = []
    for scenario_id, scenario in data_iter:
        if scenario_id in tagged_file_names:
            # Find host actor of scenario
            host_actor = 0
            for i in host_actors:
                if scenario_id in i:
                    host_actor = i[1]

            # Check if host actor is valid troughout scenario
            valid = scenario.log_trajectory.valid[host_actor]
            yaw = scenario.log_trajectory.yaw[host_actor]
            vel_x = scenario.log_trajectory.vel_x[host_actor]
            vel_y = scenario.log_trajectory.vel_y[host_actor]

            condition_1 = not False in valid                        # Ensure that the car is present during the whole scene
            condition_2 = abs(overall_angle(yaw)) < 0.1*np.pi       # Ensure the car is not making a turn
            condition_3 = avg_velocity(vel_x, vel_y) > 1.5          # Ensure the car is actually driving in the scenario
            condition_4 = np.std(yaw) > 0.03                        # Ensuring the car is not just driving straight
            condition_5 = scenario_id not in LaneChangeID           # Ensuring no double scenarios

            if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
                scenario = change_ego(scenario, host_actor)
                LaneChangeData.append(scenario)
                LaneChangeID.append(scenario_id)
                if len(LaneChangeData) == len(tagged_file_names):
                    break
            else:
                tagged_file_names.remove(scenario_id)

        if len(LaneChangeData) == len(tagged_file_names):
            break
    return LaneChangeData, LaneChangeID

