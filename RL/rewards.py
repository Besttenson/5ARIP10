from jax import numpy as jnp

def reward_calculation(simulator_state, ego_index, actions, travelled_step, stop_step, turn_factor, angle_threshold, env):
    """
    Calculate the updated reward based on the vehicle's alignment with the lane and its turning efficiency.

    This function integrates the environmental reward with dynamic adjustments based on the vehicle's current
    actions relative to the lane's direction. It rewards the vehicle for maintaining alignment with the lane and
    penalizes or rewards for turning actions based on how quickly they are executed after a decision point.

    Parameters:
        current_state : The current state object of the environment, which includes details such as
                                road graph points.
        ego_index (int): The index of the ego vehicle in the simulator state object.
        actions : An array of actions taken by the ego vehicle [velocity, yawrate].
        travelled_step (int): The total number of steps (or time units) travelled since the start or the last action.
        stop_step (int): The last step at moving straight
        turn_factor (float): A factor that scales the additional reward or penalty for turning based on the
                             responsiveness of the action.
        angle_threshold (float): The threshold for angle difference that determines whether the vehicle's
                                 action is aligned closely enough with the lane.

    Returns:
        tuple:
            - float: The updated total reward after evaluating the current step.
            - int: The updated stop_step, indicating the last step a significant directional action was taken or maintained.

    Raises:
        This function assumes that all input parameters are provided in correct form and does not handle exceptions
        internally. Ensure that 'current_state' includes 'roadgraph_points' attribute and 'actions' array is appropriately formatted.
    """

    # Fill in your own reward function here

    ego_current_pos = simulator_state.sim_trajectory.xy[ego_index, simulator_state.timestep]
    ego_log_pos = simulator_state.log_trajectory.xy[ego_index, simulator_state.timestep]

    result = jnp.sqrt(jnp.sum((ego_current_pos - ego_log_pos) ** 2))
    reward = -result

    return reward
