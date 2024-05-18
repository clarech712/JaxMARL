"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
from decimal import Decimal
import csv

import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper
from jaxmarl.environments.investment import VoteEnv

import wandb
import functools

import matplotlib.pyplot as plt
from collections import Counter
import time


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            10, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(10, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(10, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def get_rollout(runner_state, config, tail, mech_pair):
    """Get rollout with specific train state

    Returns: state_seq
    """
    env = VoteEnv(
        num_rounds=config["num_rounds"],
        num_games=config["num_games"],
        tail=tail,
        mech_pair=mech_pair
        )

    # Action space uniform for all agents
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Unpack runner state
    train_state, _, _, last_done, hstate, rng = runner_state

    # Observation space also uniform for all agents
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 10)

    # Reconstruct net
    network.init(key_a, init_hstate, init_x)
    network_params = train_state.params

    done = False

    reset_key = jax.random.split(key_r, config["NUM_ENVS"])
    obs, state = jax.vmap(env.reset, in_axes=(0,))(reset_key)
    state_seq = [state]
    while not done:
        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        ac_in = (
            obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi, _ = network.apply(network_params, hstate, ac_in)
        action = pi.sample(seed=_rng)
        env_act = unbatchify(
            action, env.agents, config["NUM_ENVS"], env.num_agents
        )
        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obs, state, _, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, state, env_act)
        done = done["__all__"][0]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, tail, mech_pair):
    # env = jaxmarl.make(config["ENV_NAME"])
    env = VoteEnv(
        num_rounds=config["num_rounds"],
        num_games=config["num_games"],
        tail=tail,
        mech_pair=mech_pair
        )
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = MPELogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 10)
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 10)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.transpose(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean(where=(1 - traj_batch.done))

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
                        entropy = pi.entropy().mean(where=(1 - traj_batch.done))

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(
                    init_hstate, (config["NUM_STEPS"], config["NUM_ACTORS"])
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :].squeeze().transpose()
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        # the metrics have an agent dimension, but this is identical
                        # for all agents so index into the 0th item of that dimension.
                        "returns": metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0]
                        ].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                    }
                )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def plot_contributions(idx, state_seq, mech, rival_mechs, gen, config):
    """Plots contributions

    Returns: N/A
    """
    # Part 1: Contributions
    rel_contribs = [state.contributions / state.agents_money for state in state_seq]
    rel_contribs_head = jnp.array([rel_contrib[:, :, :, 0] for rel_contrib in rel_contribs])
    rel_contribs_tail = jnp.array([jnp.mean(rel_contrib[:, :, :, 1:], axis=-1) for rel_contrib in rel_contribs])

    # Part 2: Mechanisms
    mechs = jnp.array([state.mech for state in state_seq])

    # Part 3: Neighbour
    nidx = (idx + 1) % config["population_size"]
    rival_mech = rival_mechs[nidx]

    # Part 4: Plot
    for idx in range(5):
        tail = (idx + 1) * 2

        plt.figure(figsize=(10, 6))
        plt.plot(range(1,len(rel_contribs_head)+1), rel_contribs_head[:, nidx, idx, 0], label='Head')
        plt.plot(range(1,len(rel_contribs_tail)+1), rel_contribs_tail[:, nidx, idx, 0], label='Tail')

        # Color background based on mech value
        for i, smech in enumerate(mechs[:, nidx, idx, 0]):
            if smech == 0:
                plt.axvspan(i+0.5, i+1.5, color='lightgreen', alpha=0.3)  # Change color and alpha as needed for mech=0

        plt.xlabel('Step')
        plt.ylabel('Relative Contribution')
        plt.title(
            f"pop_size {config['population_size']}; select_size {config['selected_size']}; num_gen {config['num_generations']}; gen {gen+1}" +
            f"; mut_rate {config['mutation_rate']}; eta {config['eta']}" +
            f"\n(green, white) ({mech}, {rival_mech}); tail {tail}"
        )
        plt.legend()

        # Save plot
        codename = f"ps{config['population_size']}_ss{config['selected_size']}_ng{config['num_generations']}_mr{config['mutation_rate']}_e{config['eta']}"
        plt.savefig(f"results/rnn/{codename}/{codename}_g{gen+1}_m{mech}_rm{rival_mech}_t{tail}.png")
        plt.close()


def get_state_seq(mech, rival_mechs, tails, config):
    """Meta-objective

    Returns: total number of votes against rival mechanisms
    """
    def process_rival(rival_mech):
        def process_tail(tail):
            mech_pair = jnp.array([mech, rival_mech])

            rng = jax.random.PRNGKey(config["SEED"])
            train_jit = jax.jit(make_train(config, tail, mech_pair), device=jax.devices()[0])
            out = train_jit(rng)

            runner_state, _ = out["runner_state"]
            state_seq = get_rollout(runner_state, config, tail, jnp.array([mech, rival_mech]))
            return state_seq

        return jax.vmap(process_tail)(jnp.array(tails))

    state_seq = jax.vmap(process_rival)(jnp.array(rival_mechs))
    return state_seq


def get_score(state_seq):
    """Gets score from state_seq

    Returns: score
    """
    mechs = [state.mech for state in state_seq]
    counts = jnp.sum(jnp.array(mechs) == jnp.array(0))
    return counts


def mutate(config, mech, key):
    """Mutates the parameters of mech
    
    Returns: mutated_mech
    """
    # Mutation params
    mutation_rate = config['mutation_rate']
    eta = config['eta']

    # Obtain v and w
    v, w = mech
    v_key, w_key = jax.random.split(key, 2)

    # Randomly decide whether to mutate each parameter with a higher rate
    v_mutate = jax.random.bernoulli(key, mutation_rate)
    w_mutate = jax.random.bernoulli(key, mutation_rate)

    # Add smaller noise to the parameters if they are mutated
    v = v + jax.random.normal(v_key, ()) * v_mutate * eta
    w = w + jax.random.normal(w_key, ()) * w_mutate * eta

    # Clip the parameters to ensure they stay within the range [0, 1]
    v = jnp.clip(v, 0, 1)
    w = jnp.clip(w, 0, 1)

    mutated_mech = (v, w)
    return mutated_mech


# Define the genetic algorithm
def genetic_algorithm(tails, config, key):
    """Simple genetic algorithm
    
    Returns: best_mech
    """
    # Retrieve algo params from config
    population_size = config["population_size"]
    selected_size = config["selected_size"]
    num_generations = config["num_generations"]

    # Create initial population
    key, pop_key = jax.random.split(key, 2)
    current_population = jax.random.uniform(pop_key, (population_size, 2))
    print(f"Current population:\n{current_population}")

    for gen in range(num_generations):
        print("********************")
        # Extend current population with rival mechanisms
        fixed_mechs = np.array([(1, 1), (0, 1), (0, 0.25)])  # Mechanisms to add
        rival_mechs = np.concatenate((current_population, fixed_mechs), axis=0)
        # state_seqs = [get_state_seq(mech, current_population, tails, config) for mech in current_population]
        state_seqs = [get_state_seq(mech, rival_mechs, tails, config) for mech in current_population]
        
        # Calculate scores for each individual in the population
        scores = jnp.array([get_score(state_seq) for state_seq in state_seqs])
        for idx, (state_seq, mech) in enumerate(zip(state_seqs, current_population)):
            plot_contributions(idx, state_seq, mech, current_population, gen, config)
        print(f"Scores: {scores}")

        # Get the parameters of the individual with the highest score
        best_idx = jnp.argmax(scores)
        best_mech = current_population[best_idx]
        best_score = scores[best_idx]
        print(f"Best score: {best_score}")

        # Perform selection based on score
        selected_indices = jnp.argsort(scores)[-selected_size:]
        selected_population = current_population[selected_indices]
        print(f"Selected population:\n{selected_population}")

        next_generation = []
        for _ in range(population_size):
            # Randomly select two parents from the selected population
            key, par_key, child_key = jax.random.split(key, 3)
            parent1, parent2 = jax.random.choice(par_key, selected_population, (2,), replace=False)

            # Create a child by combining and mutating the parents' parameters
            child = mutate(config, (parent1[0], parent2[1]), child_key)
            next_generation.append(child)

        current_population = jnp.array(next_generation)
        print(f"Next generation:\n{current_population}")

    return best_mech


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_pit")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"]
    )

    tails = jnp.array([2, 4, 6, 8, 10])
    key = jax.random.PRNGKey(0)

    start_time = time.time()
    best_mech = genetic_algorithm(
        tails,
        config,
        key
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Best mechanism: {best_mech}")
    print(f"Time taken: {elapsed_time} seconds")


if __name__ == "__main__":
    main()
