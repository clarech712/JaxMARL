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


class InvestmentEnv(MultiAgentEnv):
    """Represents investment environment
    """

    def __init__(
            self,
            num_agents=4,
            num_rounds=10,
            tail=2,
            seed=0,
            v=1,
            w=1
            ):
        super().__init__(num_agents=4)
        key = jax.random.PRNGKey(seed)

        # Agents
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Rounds
        self.num_rounds = num_rounds
        self.tail = tail

        # Endowments
        # the amount of money a player receives each round
        self.head_idx = 0 # TODO: Equivalent to head = jax.random.choice(key, jnp.arange(self.num_agents))?
        self.endowments = jnp.repeat(self.tail, repeats=self.num_agents)
        self.endowments = self.endowments.at[self.head_idx].set(10)

        # Action spaces
        self.action_spaces = {a: Discrete(11) for a, e in zip(self.agents, self.endowments)}

        # Observation spaces
        self.observation_spaces = {
            a: MultiDiscrete([10] * (3 * self.num_agents + 1))
            for a in self.agents
            }

        # Manifold
        self.v = v
        self.w = w
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
            step=1
            )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key, state, actions):
        """Performs step transitions in the environment

        Returns: obs, state, rewards, done, info
        """
        # Get the actions as array
        actions = jnp.array([actions[i] for i in self.agents]).reshape((self.num_agents,))
        actions = jnp.minimum(actions, self.endowments) # actions % (self.endowments + 1)

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
        y_abs = self.r * (self.w * actions + (1 - self.w) * jnp.mean(other_rewards, axis=1))
        y_rel = self.r * (common_pot / (tot_ratio + 1e-6)) * (self.w * ro + (1 - self.w) * jnp.mean(other_ratios, axis=1)) # TODO should these be a mean??
        y = self.v * y_rel + (1 - self.v) * y_abs

        # Find rewards
        payouts = y - actions + self.endowments
        rewards = {a: payouts[i] for a, i in zip(self.agents, range(self.num_agents))}

        # Update the environment state
        step = state.step + 1
        state = State(
            agents_money=self.endowments,
            contributions=actions,
            payouts=payouts,
            step=step
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
            ).astype(jnp.int32)

        # Tell agent if head or tail
        roles = jnp.repeat(0, repeats=self.num_agents)
        roles = roles.at[self.head_idx].set(1)

        return {a: jnp.concatenate((obs, jnp.array([role]))).astype(jnp.float32) for a, role in zip(self.agents, roles)}

    def is_terminal(self, state):
        """Check whether state is terminal

        Returns: is_terminal
        """
        is_terminal = state.step > self.num_rounds # + 1

        return is_terminal

    def render(self, state):
        """Renders environment on command line

        Returns: N/A
        """
        print(f"\nCurrent step: {state.step}")
        print(f"Agents' money: {state.agents_money}")
        print(f"Contributions: {state.contributions}")
        print(f"Payouts: {state.payouts}")


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

    env = InvestmentEnv()

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
