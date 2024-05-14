"""
Investment environment based on https://doi.org/10.1038/s41562-022-01383-x
"""
from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp
import chex
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete, MultiDiscrete


@struct.dataclass
class State:
    """State of investment game
    """

    agents_money: chex.Array
    contributions: chex.Array
    payouts: chex.Array
    step: jnp.float32
    mech: int


class VoteEnv(MultiAgentEnv):
    """Represents investment environment
    """

    def __init__(
            self,
            num_agents=4,
            num_rounds=5,
            num_games=3,
            tail=2,
            seed=0,
            mech_pair=jnp.array([(1, 1), (0, 1)])
            ):
        super().__init__(num_agents=4)
        key = jax.random.PRNGKey(seed)

        # Agents
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Rounds
        self.num_rounds = num_rounds
        self.num_games = num_games
        self.tail = tail

        # Endowments
        # the amount of money a player receives each round
        head = 0 # TODO: Equivalent to head = jax.random.choice(key, jnp.arange(self.num_agents))?
        self.endowments = jnp.repeat(self.tail, repeats=self.num_agents)
        self.endowments = self.endowments.at[head].set(10)

        # Action spaces
        self.action_spaces = {a: Discrete(22) for a, e in zip(self.agents, self.endowments)}

        # Observation spaces
        self.observation_spaces = {
            a: MultiDiscrete([10] * (3 * self.num_agents))
            for a in self.agents
            }

        # Manifold
        self.mech_pair = mech_pair
        self.r = 1.6

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key):
        """Performs resetting of environment

        Returns: obs, state
        """
        state = State(
            agents_money=self.endowments,
            contributions=jnp.zeros(self.num_agents, dtype=jnp.int32),
            payouts=jnp.zeros(self.num_agents, dtype=jnp.float32),
            step=1,
            mech=0
            )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key, state, actions):
        """Performs step transitions in the environment

        Returns: obs, state, rewards, done, info
        """
        # Select mechanism for round
        def vote():
            """Votes for mechanism

            Returns: mechanism index
            """
            votes = jnp.array([actions[i] // 11 for i in self.agents]).reshape((self.num_agents,))
            return jnp.argmax(jnp.bincount(votes, length=2))

        mech = jax.lax.cond(state.step % self.num_rounds == 0, vote, lambda: state.mech)
        v, w = self.mech_pair[mech]

        # Get the actions as array
        actions = jnp.array([actions[i] % 11 for i in self.agents]).reshape((self.num_agents,))
        actions = jnp.minimum(actions, self.endowments)

        # Common pot
        common_pot = jnp.sum(actions)

        # Find ratios
        contribution_ratios = actions / state.agents_money
        tot_ratio = common_pot / jnp.sum(state.agents_money)

        # Find other rewards
        other_rewards = jnp.repeat(actions.reshape([1, self.num_agents]), self.num_agents, axis=0).reshape(self.num_agents, self.num_agents)
        di = jnp.diag_indices(self.num_agents)
        other_rewards = other_rewards.at[di].set(0)

        # Find other ratios
        ro = actions / self.endowments
        other_ratios = jnp.repeat(contribution_ratios.reshape([1, self.num_agents]), self.num_agents, axis=0).reshape(self.num_agents, self.num_agents)
        other_ratios = other_ratios.at[di].set(0)

        # Find y
        y_abs = self.r * (w * actions + (1 - w) * jnp.mean(other_rewards, axis=1))
        y_rel = self.r * (common_pot / (tot_ratio + 1e-6)) * (w * ro + (1 - w) * jnp.mean(other_ratios, axis=-1)) # TODO: Should these be a mean??
        y = v * y_rel + (1 - v) * y_abs

        # Find rewards
        payouts = y - actions + self.endowments
        rewards = {a: payouts[i] for a, i in zip(self.agents, range(self.num_agents))}

        # Update the environment state
        step = state.step + 1
        state = State(
            agents_money=self.endowments,
            contributions=actions,
            payouts=payouts,
            step=step,
            mech=mech
            )

        # Prepare remaining outputs
        obs = self.get_obs(state)
        done = self.is_terminal(state)
        dones = {a: done for a in self.agents + ["__all__"]}
        info = {}

        return (obs, state, rewards, dones, info)

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state):
        """Applies observation function to state.

        Returns: obs
        """
        obs = jnp.concatenate(
            (state.agents_money, state.contributions, state.payouts)
        ).astype(jnp.float32)

        return {a: obs for a in self.agents}

    def is_terminal(self, state):
        """Check whether state is terminal

        Returns: is_terminal
        """
        is_terminal = state.step > (self.num_rounds * self.num_games)

        return is_terminal

    def render(self, state):
        """Renders environment on command line

        Returns: N/A
        """
        print(f"\nCurrent step: {state.step}")
        print(f"Agents' money: {state.agents_money}")
        print(f"Contributions: {state.contributions}")
        print(f"Payouts: {state.payouts}")
        print(f"Mechanism: {state.mech}")


    @property
    def name(self) -> str:
        """Environment name

        Returns: env_name
        """
        return "EconomicsEnv"

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError


def example():
    """Simple manipulations of environment for debugging

    Returns: N/A
    """
    key = jax.random.PRNGKey(0)

    env = VoteEnv()

    _, state = env.reset(key)

    while not env.is_terminal(state):
        key, key_step, key_act = jax.random.split(key, 3)

        # Render environment state
        env.render(state)

        # Sample random actions
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            a: env.action_spaces[a].sample(key_act[i])
            for i, a in enumerate(env.agents)
            }

        # Print action
        print("Actions:", actions)

        # Perform step in environment
        obs, state, rewards, done, _ = env.step_env(key_step, state, actions)

        # Print metadata
        print(f"Observation: {obs}")
        print(f"Rewards: {rewards}")
        print(f"Done: {done}")


if __name__ == "__main__":
    example()
