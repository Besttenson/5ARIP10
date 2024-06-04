import abc
from typing import Callable, Sequence, TypeVar

import chex
import jax
import jax.numpy as jnp
from waymax import datatypes

ActorState = datatypes.PyTree
Params = datatypes.PyTree
Action = datatypes.PyTree

@chex.dataclass(frozen=True)
class WaymaxActorOutput:
  """Output of the Waymax actor including an action and its internal state.

  Attributes:
    actor_state: Internal state for whatever the agent needs to keep as its
      state. This can be recurrent embeddings or accounting information.
    action: Action of shape (..., num_objects) predicted by the Waymax actor at
      the most recent simulation step given the inputs in the `select_action`
      function of `WaymaxActorCore`.
    is_controlled: A binary indicator of shape (..., num_objects) representing
      which objects are controlled by the actor.
  """

  actor_state: ActorState
  action: Action
  is_controlled: jax.Array

  def validate(self):
    """Validates shapes."""
    chex.assert_equal_shape([self.action, self.is_controlled])


class WaymaxActorCore(abc.ABC):
  """Interface that defines actor functionality for inference."""

  @abc.abstractmethod
  def init(self, rng: jax.Array, state: datatypes.SimulatorState) -> ActorState:
    """Initializes the actor's internal state.

    ActorState is a generic type which can contain anything that the agent
    needs to pass through to the next call, e.g. for recurrent state or
    batch normalization. The `init` function takes a random key to help
    randomize initialization and the initial timestep.

    Args:
      rng: A random key.
      state: The initial simulator state.

    Returns:
      The actor's initial state.
    """

  @abc.abstractmethod
  def select_action(
      self,
      action: list,
      state: datatypes.SimulatorState,
      actor_state: ActorState,
      rng: jax.Array,
  ) -> WaymaxActorOutput:
    """Selects an action given the current simulator state.

    Args:
      params: Actor parameters, e.g. neural network weights.
      state: The current simulator state.
      actor_state: The actor state, e.g. recurrent state or batch normalization.
      rng: A random key.

    Returns:
      An actor output containing the next action and actor state.
    """

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Name of the agent used for inspection and logging."""


_ActorCore = TypeVar('_ActorCore', bound=WaymaxActorCore)


def register_actor_core(
    actor_core_cls: type[_ActorCore],
) -> type[_ActorCore]:
  """Registers an ActorCore class as a PyTree node."""
  jax.tree_util.register_pytree_node(
      actor_core_cls,
      flatten_func=lambda self: (tuple(), self),
      unflatten_func=lambda self, children: self,
  )
  return actor_core_cls

def actor_core_factory_customized(
    init: Callable[[jax.Array, datatypes.SimulatorState], ActorState],
    select_action: Callable[
        [
            list,
            datatypes.SimulatorState,
            ActorState,
            jax.Array,
        ],
        WaymaxActorOutput,
    ],
    name: str = 'WaymaxActorCore',
) -> WaymaxActorCore:
  """Creates a WaymaxActorCore from pure functions.

  Args:
    init: A function that initializes the actor's internal state. This is a
      generic type which can contain anything that the agent needs to pass
      through to the next call. The `init` function takes a random key to help
      randomize initialization and the initial timestep. It should return its
      specific internal state.
    select_action: A function that selects an action given the current simulator
      state of the environment, the previous actor state and an optional random
      key. Returns the action and the updated internal actor state.
    name: Name of the agent used for inspection and logging.

  Returns:
    An actor core instance defined by init and select_action.
  """

  @register_actor_core
  class ActorCoreInstance(WaymaxActorCore):
    """A generic WaymaxActorCore instance."""

    def init(
        self, rng: jax.Array, state: datatypes.SimulatorState
    ) -> ActorState:
      return init(rng, state)

    def select_action(
        self,
        action: list,
        state: datatypes.SimulatorState,
        actor_state: ActorState,
        rng: jax.Array,
    ) -> WaymaxActorOutput:
      return select_action(action, state, actor_state, rng)

    @property
    def name(self) -> str:
      return name

  return ActorCoreInstance()