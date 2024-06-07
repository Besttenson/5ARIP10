# Do all needed imports
import numpy as np
import mediapy
import dataclasses
import os
import jax
from jax import random
from jax import numpy as jnp
import tensorflow as tf
import json
from waymax import config as _config
from waymax.config import DatasetConfig, DataFormat
from waymax import dataloader
from waymax import env as _env
from waymax import datatypes
from waymax import visualization

# Define a data config with locations of WOMD files
DATA_LOCAL_01 = DatasetConfig(
    path= ''# LOCATION OF DIRECTORY CONTAINING WOMD TFRECORD FILES + 'training_tfexample.tfrecord@XX'
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
)

data_config = DATA_LOCAL_01
data_config = dataclasses.replace(data_config, max_num_objects=32)

# Custom dataloader that loads scenario IDs.
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
    # Define the location where the categorization files are located
    folder_path = ''# LOCATION OF DIRECTORY CONTAINING THE CATEGORIZED _SCXX.SJON FILES
    tagged_file_names = []
    host_actors = []

    # Iterate over each file in the json folder and determine the scenario_id and host actor (actor that makes the lane change)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            with open(folder_path + "\\" + file_name, 'r', encoding='utf-8') as file:
                file_content = json.loads(file.read())
                host_actor_numbers = [value["host_actor"] for value in file_content.values()]
                scenario_id = file_name.split("_")
                tagged_file_names.append(scenario_id[2])
                host_actors.append((scenario_id[2], host_actor_numbers[0]))
    return tagged_file_names, host_actors

# Calculate the overall angle change of the vehicle
def overall_angle(yaw):
    angle_change = yaw[0] - yaw[-1]
    return(angle_change)

# Calculate the absolute velocity of the vehicle
def avg_velocity(x_velocities, y_velocities):
    velocity_magnitude = 0
    for i in range(len(x_velocities)):
        velocity_magnitude += np.sqrt(x_velocities[i]**2 + y_velocities[i]**2)
    avg_velocity_magnitude = velocity_magnitude/len(x_velocities)
    return avg_velocity_magnitude

# Change the ego vehicle of the given state to the desired ego vehicle
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
    # Perform extra filtering on the scenarios
    LaneChangeData = []
    LaneChangeID = []
    for scenario_id, scenario in data_iter:
        if scenario_id in tagged_file_names:
            # Find host actor that belongs to this scenario
            host_actor = 0
            for i in host_actors:
                if scenario_id in i:
                    host_actor = i[1]
            
            # Obtain the relevant data about the host actor
            valid = scenario.log_trajectory.valid[host_actor]
            yaw = scenario.log_trajectory.yaw[host_actor]
            vel_x = scenario.log_trajectory.vel_x[host_actor]
            vel_y = scenario.log_trajectory.vel_y[host_actor]

            # Specify the conditions that need to be satisfied to pass filtering
            condition_1 = not False in valid                        # Ensure that the car is present during the whole scene
            condition_2 = abs(overall_angle(yaw)) < 0.1*np.pi       # Ensure the car is not making a turn
            condition_3 = avg_velocity(vel_x, vel_y) > 1.5          # Ensure the car is actually driving in the scenario
            condition_4 = np.std(yaw) > 0.03                        # Ensuring the car is not just driving straight
            condition_5 = scenario_id not in LaneChangeID           # Ensuring no double scenarios

            # Change the ego vehicle of the scenario, print and store the scenario data
            if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
                print(scenario_id, host_actor)
                scenario = change_ego(scenario, host_actor)
                LaneChangeData.append(scenario)
                LaneChangeID.append(scenario_id)   
                # Break if all scenarios are found     
                if len(LaneChangeData) == len(tagged_file_names):
                    break
            # Remove the filenames of scenarios that did not pass the extra filtering
            else:
                tagged_file_names.remove(scenario_id)
        # Break if all scenarios are found
        if len(LaneChangeData) == len(tagged_file_names):
            break
    return LaneChangeData, LaneChangeID

if __name__ == "__main__":
    start_time = time.time()
    data_iter = decode_bytes(dataloader.get_data_generator(data_config, _preprocess, _postprocess))
    tagged_file_names, host_actors = categorized_scenarios()
    LaneChangeData, LaneChangeID = extra_filtering(tagged_file_names, host_actors, data_iter)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time to run scenario filtering:", elapsed_time, "[seconds]")
    print("Correct lane change IDs:", LaneChangeID)