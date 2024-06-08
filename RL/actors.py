from typing import Callable
import jax
import jax.numpy as jnp
from waymax.agents import actor_core
from waymax import datatypes
from waymax import dynamics
from utils import pi_to_pi
from abstract_actor import actor_core_factory_customized

def create_customized_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array]
) -> actor_core.WaymaxActorCore:
  """Creates an actor with constant speed without changing objects' heading.

  Note the difference against ConstantSpeedPolicy is that an actor requires
  input of a dynamics model, while a policy does not (it assumes to use
  StateDynamics).

  Args:
    dynamics_model: The dynamics model the actor is using that defines the
      action output by the actor.
    is_controlled_func: Defines which objects are controlled by this actor.

  Returns:
    An statelss actor that drives the controlled objects with constant speed.
  """

  def select_action(  # pytype: disable=annotation-type-mismatch
      action: list,
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: jax.Array = None,
  ) -> actor_core.WaymaxActorOutput:
    """Computes the actions using the given dynamics model and speed."""
    del  actor_state, rng  # unused.
    traj_t0 = datatypes.dynamic_index(
        state.sim_trajectory, state.timestep, axis=-1, keepdims=True
    )
    speed = action[0]
    velocity = speed
    yawrate = action[1]
    new_yaw = pi_to_pi(traj_t0.yaw + yawrate*datatypes.TIME_INTERVAL)
    # new_yaw = yaw

    is_controlled = is_controlled_func(state)
    traj_t1 = traj_t0.replace(
        x=traj_t0.x + velocity * datatypes.TIME_INTERVAL * jnp.cos(new_yaw),
        y=traj_t0.y + velocity * datatypes.TIME_INTERVAL * jnp.sin(new_yaw),
        vel_x=traj_t0.vel_x + velocity * jnp.cos(new_yaw),
        vel_y=traj_t0.vel_y + velocity * jnp.sin(new_yaw),
        yaw = new_yaw,
        valid=is_controlled[..., jnp.newaxis] & traj_t0.valid,
        timestamp_micros=(
            traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL
        ),
    )

    traj_combined = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=-1), traj_t0, traj_t1
    )
    actions = dynamics_model.inverse(
        traj_combined, state.object_metadata, timestep=0
    )

    # Note here actions' valid could be different from is_controlled, it happens
    # when that object does not have valid trajectory from the previous
    # timestep.
    return actor_core.WaymaxActorOutput(
        actor_state=None,
        action=actions,
        is_controlled=is_controlled,
    )

  return actor_core_factory_customized(
      init=lambda rng, init_state: None,
      select_action=select_action,
      name='ego_customized_actor',
  )