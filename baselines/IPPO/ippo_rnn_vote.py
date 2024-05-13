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

def get_rollout(runner_state, config):
    """Get rollout with specific train state

    Returns: state_seq
    """
    env = VoteEnv(
        num_rounds=config["num_rounds"],
        num_games=config["num_games"],
        tail=config["tail"],
        mech_pair=jnp.array([(config["v_0"], config["w_0"]), (config["v_1"], config["w_1"])])
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
    start_net_init = time.time()
    network.init(key_a, init_hstate, init_x)
    net_init_time = time.time() - start_net_init
    print(f"Network initialization time: {net_init_time} seconds")

    network_params = train_state.params

    done = False

    reset_key = jax.random.split(key_r, config["NUM_ENVS"])
    obs, state = jax.vmap(env.reset, in_axes=(0,))(reset_key)
    state_seq = [state]

    start_loop = time.time()
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
    loop_time = time.time() - start_loop
    print(f"Loop time: {loop_time} seconds")

    return state_seq



def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    # env = jaxmarl.make(config["ENV_NAME"])
    env = VoteEnv(
        num_rounds=config["num_rounds"],
        num_games=config["num_games"],
        tail=config["tail"],
        mech_pair=jnp.array([(config["v_0"], config["w_0"]), (config["v_1"], config["w_1"])])
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
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
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
            metric = jax.tree_map(
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


def plot_contributions(state_seq, config):
    """Plots contributions

    Returns: N/A
    """
    # Extract contributions for each round
    contributions = jnp.array([state.contributions[0] for state in state_seq]) # TODO: Compatible with pit

    # Separate contributions for head and tail
    head_idx = np.where(state_seq[0].agents_money[0] == 10)[0][0]
    tail_idx = (head_idx + 1) % len(state_seq[0].agents_money[0])
    tail_endowment = state_seq[0].agents_money[0][tail_idx]
    head_contributions = [(contribution[head_idx] / 10) for contribution in contributions]
    tail_contributions = [
        [(contribution[i] / tail_endowment) for contribution in contributions]
        for i in range(len(state_seq[0].agents_money[0])) if i != head_idx
    ]

    # Calculate average contribution for each round
    avg_contributions = np.mean(tail_contributions, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,len(head_contributions)+1), head_contributions, label='Head')
    plt.plot(range(1,len(avg_contributions)+1), avg_contributions, label='Tail')
    plt.xlabel('Round')
    plt.ylabel('Relative contribution')
    
    total_timesteps = format_e(Decimal(str(config['TOTAL_TIMESTEPS'])))
    title = (
        f"Relative contribution per round of gameplay after training ({config['num_games']} games)\n" +
        f"{config['experiment_name']}; tail {config['tail']}; seed {config['SEED']}; timesteps {total_timesteps}; {config['num_games']} games of {config['num_rounds']} rounds each"
    )
    plt.title(title)    
    plt.legend()

    # Save plot
    plt.savefig(f"results/rnn/strategy/{config['experiment_name']}_{config['tail']}.png")


def plot_mechs(state_seq, config):
    """Plots mechs
    
    Returns: N/A
    """
    # Extract game played for each round
    mechs = [state.mech[0].item() for state in state_seq] # TODO: Compatible with pit

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,len(mechs)+1), mechs, label='Mechanism')
    plt.xlabel('Round')
    plt.ylabel('Mechanism')

    total_timesteps = format_e(Decimal(str(config['TOTAL_TIMESTEPS'])))
    title = (
        f"Mechanism chosen per round of gameplay after training ({config['num_games']} games)\n" +
        f"{config['experiment_name']}; tail {config['tail']}; seed {config['SEED']}; timesteps {total_timesteps}; {config['num_games']} games of {config['num_rounds']} rounds each"
    )
    plt.title(title)
    plt.legend()

    # Save plot
    plt.savefig(f"results/rnn/mech/{config['experiment_name']}_{config['tail']}.png")

    # Count occurrences of each element
    counts = jnp.sum(jnp.array(mechs) == jnp.array(0))
    return counts


def plot_returns(returned_episode_returns, config):
    """Plots returns

    Returns: N/A
    """
    # Determine mean returns
    mean_returns = returned_episode_returns.mean(-1).reshape(-1)
    assert(jnp.sum(jnp.isnan(mean_returns)) == 0)

    # Plot
    x = np.arange(len(mean_returns))
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_returns)
    plt.xlabel("Round")
    plt.ylabel("Average return")
    total_timesteps = format_e(Decimal(str(config['TOTAL_TIMESTEPS'])))
    title = (
        "Average return across agents per training round\n" +
        f"{config['experiment_name']}; tail {config['tail']}; seed {config['SEED']}; timesteps {total_timesteps}"
    )
    plt.title(title)

    # Save plot
    plt.savefig(f"results/rnn/train/{config['experiment_name']}_{config['tail']}.png")


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_vote")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"]
    )
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config), device=jax.devices()[0])

    start = time.time()
    out = train_jit(rng)
    train_time = time.time() - start
    print(f"Training time: {train_time} seconds")

    returned_episode_returns = out["metrics"]["returned_episode_returns"]
    plot_returns(returned_episode_returns, config)

    start = time.time()
    runner_state, _ = out["runner_state"]
    state_seq = get_rollout(runner_state, config)
    rollout_time = time.time() - start
    print(f"Rollout time: {rollout_time} seconds")
    plot_contributions(state_seq, config)

    start = time.time()
    score = plot_mechs(state_seq, config)
    plot_time = time.time() - start
    print(f"Plotting time: {plot_time} seconds")

    print(f"Score: {score}")


if __name__ == "__main__":
    main()
