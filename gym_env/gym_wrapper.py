import jax
import jax.numpy as jnp
from waymax import env as _env
from waymax import dynamics
from waymax import agents
import gym
from gym import spaces
import numpy as np
from waymax.datatypes import observation
from waymax.datatypes import roadgraph
from waymax import datatypes
from Config import Config
from rewards import reward_calculation
from actors import create_customized_actor

class WaymaxWrapper(gym.Env):
    """
    Provides a `"world_state"` observation for the centralized critic.
    world state observation of dimension: (num_agents, world_state_size)
    """

    def __init__(self, env: _env.MultiAgentEnvironment, scenario, config: Config, obs_with_agent_id=False):
        super(WaymaxWrapper, self).__init__()
        self._env = env
        self.num_agents = self._env.config.max_num_objects
        self.obs_with_agent_id = obs_with_agent_id
        self.scenario = scenario
        self.env_state = self._env.reset(scenario)
        self.scenario_total_steps = self.env_state.remaining_timesteps
        self._travelled_steps = 0
        self.stop_step = 0
        self.dynamics_model = dynamics.InvertibleBicycleModel(
        normalize_actions=True,)  # This means we feed in all actions as in [-1, 1]
        self.obj_idx = jnp.arange(self._env.config.max_num_objects)
        index = self.env_state.object_metadata.is_sdc
        self.config = config

        for num, idx in enumerate(index):
          if idx:
            self.ego_index = num
            break
          else:
            self.ego_index = 0
        self.sim_actors = agents.create_expert_actor(dynamics_model = self.dynamics_model, is_controlled_func=lambda state: self.obj_idx != self.ego_index)
        self.ego_actor = create_customized_actor(dynamics_model = self.dynamics_model, is_controlled_func=lambda state: self.obj_idx == self.ego_index)

        # Size of each agent's observation (including other agents and rg points)
        self._agent_obs_size = self.num_agents * 7 + env.config.observation.roadgraph_top_k * 7
        
        if config.mode == 'simple':
           self.observation_space = spaces.Dict({
            "time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=jnp.float32),
            "coordinates": spaces.Box(low=-np.inf, high=np.inf, shape=(5, 2), dtype=jnp.float32)
        })
        else: 
            self.observation_space = spaces.Dict({"ego": spaces.Box(low=-np.inf, high=np.inf, shape=(self._agent_obs_size,), dtype=jnp.float32)})
        


        # self.action_space = spaces.Box(low=-20.0, high=20.0,shape=(2,), dtype=np.float32)
        action_space_low = np.array(config.action_low, dtype=np.float32)
        action_space_high = np.array(config.action_high, dtype=np.float32)
        action_space_shape = action_space_low.shape
        action_space_dtype = np.float32

        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, shape=action_space_shape, dtype=action_space_dtype)

        # Lane changing variables
        self.next_heading_change_increment = 0
        self.lane_changing_time = 0
        self.time_on_lane = 0
        self.lane_change_complete = False
        self.total_heading_change = 0

    def get_obs(self, env_state):
        if self.config.mode == 'simple':
            obs = env_state.sim_trajectory.xy[:5, env_state.timestep]
        else:
            
            global_traj = datatypes.dynamic_index(env_state.sim_trajectory, env_state.timestep, axis=-1, keepdims=True)
            obs = {}

            # for idx in range(self._env.config.max_num_objects):
            idx = self.ego_index
            global_rg = roadgraph.filter_topk_roadgraph_points(
                env_state.roadgraph_points,
                env_state.sim_trajectory.xy[idx, env_state.timestep],
                topk=self._env.config.observation.roadgraph_top_k,
            )

            pose = observation.ObjectPose2D.from_center_and_yaw(
                xy=env_state.sim_trajectory.xy[idx, env_state.timestep],
                yaw=env_state.sim_trajectory.yaw[idx, env_state.timestep],
                valid=env_state.sim_trajectory.valid[idx, env_state.timestep],
            )

            sim_traj = observation.transform_trajectory(global_traj, pose)
            exp_rg = observation.transform_roadgraph_points(global_rg, pose)

            agent_obs = jnp.concatenate((
                sim_traj.xyz.reshape(-1), sim_traj.yaw.reshape(-1),
                sim_traj.vel_xy.reshape(-1), sim_traj.vel_yaw.reshape(-1),
                exp_rg.xyz.reshape(-1), exp_rg.dir_xyz.reshape(-1),
                exp_rg.types.reshape(-1),
            ))
            obs[f'object_{idx}'] = agent_obs

        return obs

    def reset(self):
        key = jax.random.PRNGKey(0)
        self.env_state = self._env.reset(self.scenario, key)

        self._travelled_steps = 0
        if self.config.mode == 'simple':
            obs = {
                "time": self._travelled_steps,
                "coordinates": self.get_obs(self.env_state)
            }
        else:
            obs = self.get_obs(self.env_state)
        self.stop_step = 0

        # Lane changing variables
        self.next_heading_change_increment = 0
        self.lane_changing_time = 0
        self.time_on_lane = 0
        self.lane_change_complete = False
        self.total_heading_change = 0

        return obs

    def step(self, action: list):
        
        current_state = self.env_state  # Assume env_state is stored as an attribute
        actors = [self.ego_actor, self.sim_actors]
        jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]
        outputs = [jit_select_action(action, current_state, None, None) for jit_select_action in jit_select_action_list]

        action_update = agents.merge_actions(outputs)
        env_state = self._env.step(current_state, action_update)

        reward = self._env.reward(current_state, action_update)

        self.env_state = env_state

        if self.config.mode == 'simple':
            reward = self.calculate_lane_changing_reward(current_state, action)
            obs = {
                "time": self._travelled_steps,
                "coordinates": self.get_obs(env_state)
            }
        else:
            obs = self.get_obs(env_state)
            waymax_reward = self._env.reward(current_state, action_update)
            extra_reward = reward_calculation(current_state, self.ego_index, action, self._travelled_steps, self.stop_step, 1, 20, self._env)
            reward = waymax_reward + extra_reward
        
        if self._travelled_steps == self.scenario_total_steps:
            dones = True
            self._travelled_steps = 0
        else:
            dones = False
        infos = {}

        self._travelled_steps += 1

        return obs, reward, dones, infos
    

    def calculate_lane_changing_reward(self, current_state, action):
        self.time_on_lane += 1
        lane_changing_reward = 0

        if self.time_on_lane >= 10:
            if self.config.safety_check:
                # Check if it's safe to change lanes
                current_observation = self.get_obs(current_state)

                ego_state = current_observation["ego_veh_state"]
                sim_veh_states = current_observation["sim_veh_states"]
                road_graph = current_observation["road_graph"]
                safe_change = self.is_safe_lane_change_left(ego_state, sim_veh_states, road_graph)
                
            else:
                safe_change = True
            

            if safe_change and self.lane_changing_time <= 15:
                self.lane_changing_time += 1
                self.lane_change_complete = False
                lane_changing_reward = 5

                if action[1] > jnp.pi / 3:
                    lane_changing_reward = 5
                elif action[1] > jnp.pi / 4:
                    lane_changing_reward = 3
                elif action[1] > jnp.pi / 6:
                    lane_changing_reward = 1
                else:
                    lane_changing_reward = -5

                if self.lane_changing_time == 15:  # Lane change complete
                    self.time_on_lane = 0
                    self.lane_changing_time = 0
                    self.next_heading_change_increment = 0
                    lane_changing_reward += 5
                    self.lane_change_complete = True

        if self.lane_change_complete and self.time_on_lane < 30:
            if action[1] > -jnp.pi / 3:
                lane_changing_reward = 2
            elif action[1] > -jnp.pi / 4:
                lane_changing_reward = 1.5
            elif action[1] > -jnp.pi / 6:
                lane_changing_reward = 1
            else:
                lane_changing_reward = -2

        return lane_changing_reward

    def is_safe_lane_change_left(self, ego_state=None, sim_veh_states=None, road_graph=None):
        if road_graph is None:
            return True

        left_lane_types = road_graph[2][1]  # Get types of roadpoints in the lane to the left
        if not all(type == 1 for type in left_lane_types):  # check if left lane is of type MapElementIds.LANE_FREEWAY
            return False  # Lane on the left is not a freeway

        left_lane_x = road_graph[0][1]  # Get x-coordinates of roadpoints in the lane to the left
        left_lane_y = road_graph[1][1]  # Get y-coordinates of roadpoints in the lane to the left

        distances = [((ego_state[0] - veh_state[0]) ** 2 + (ego_state[1] - veh_state[1]) ** 2) ** 0.5
                     for veh_state in sim_veh_states]

        if any((abs(veh_state[0] - x) < 1 and abs(veh_state[1] - y) < 1) for veh_state, x, y in zip(sim_veh_states, left_lane_x, left_lane_y)):
            return False  # There is a vehicle on the lane to the left

        return True
