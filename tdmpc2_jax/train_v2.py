import os
from collections import defaultdict
from functools import partial

import flax.linen as nn
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import tqdm
from flax.metrics import tensorboard
from flax.training.train_state import TrainState
from flashbax.buffers.trajectory_buffer import make_trajectory_buffer

from tdmpc2_jax import TDMPC2, WorldModel
from tdmpc2_jax.common.activations import mish, simnorm
from tdmpc2_jax.envs.dmcontrol import make_dmc_env
from tdmpc2_jax.networks import NormedLinear

# Transition namedtuple-style dict keys expected by flashbax
# We store: obs, action, reward, next_obs, terminated, truncated
# FlashBax trajectory buffer stores sequences of length (horizon+1)
# so we add transitions one step at a time with add_batch_size=num_envs

jax.config.update("jax_log_compiles", True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
    env_config = cfg['env']
    encoder_config = cfg['encoder']
    model_config = cfg['world_model']
    tdmpc_config = cfg['tdmpc2']

    ##############################
    # Logger setup
    ##############################
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    writer.hparams(cfg)

    ##############################
    # Environment setup
    ##############################
    def make_env(env_config, seed):
        def make_gym_env(env_id, seed):
            env = gym.make(env_id)
            env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.Autoreset(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        if env_config.backend == "gymnasium":
            return make_gym_env(env_config.env_id, seed)
        elif env_config.backend == "dmc":
            env = make_dmc_env(env_config.env_id, seed, env_config.dmc.obs_type)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.Autoreset(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        else:
            raise ValueError("Environment not supported:", env_config)

    if env_config.asynchronous:
        vector_env_cls = gym.vector.AsyncVectorEnv
    else:
        vector_env_cls = gym.vector.SyncVectorEnv

    env = vector_env_cls(
        [
            partial(make_env, env_config, seed)
            for seed in range(cfg.seed, cfg.seed + env_config.num_envs)
        ]
    )
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    ##############################
    # Agent setup
    ##############################
    dtype = jnp.dtype(model_config.dtype)
    rng, model_key, encoder_key = jax.random.split(rng, 3)
    encoder_module = nn.Sequential(
        [
            NormedLinear(encoder_config.encoder_dim, activation=mish, dtype=dtype)
            for _ in range(encoder_config.num_encoder_layers - 1)
        ] + [
            # NormedLinear(model_config.latent_dim, activation=None, dtype=dtype)
            nn.Dense(model_config.latent_dim,)
        ]
    )


    if encoder_config.tabulate:
        print("Encoder")
        print("--------------")
        print(
            encoder_module.tabulate(
                jax.random.key(0),
                env.observation_space.sample(),
                compute_flops=True
            )
        )

    ##############################
    # Replay buffer setup (FlashBax)
    ##############################
    dummy_obs, _ = env.reset()
    dummy_action = env.action_space.sample()
    dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(dummy_action)

    observation_size = dummy_obs.shape[-1]
    action_size = dummy_action.shape[-1]
    horizon = tdmpc_config.get('horizon', 3)

    # Dummy transition for buffer init — single env, single step
    # FlashBax stores (obs, action, reward, next_obs, terminated, truncated)
    # add_batch_size = num_envs, sample_sequence_length = horizon+1
    dummy_transition = {
        'obs': jnp.zeros((observation_size,), dtype=jnp.float32),
        'action': jnp.zeros((action_size,), dtype=jnp.float32),
        'reward': jnp.zeros((), dtype=jnp.float32),
        'next_obs': jnp.zeros((observation_size,), dtype=jnp.float32),
        'terminated': jnp.zeros((), dtype=jnp.float32),
        'truncated': jnp.zeros((), dtype=jnp.float32),
    }

    replay_buffer = make_trajectory_buffer(
        max_length_time_axis=cfg.buffer_size // env_config.num_envs,
        min_length_time_axis=horizon + 1,
        sample_batch_size=tdmpc_config.batch_size,
        add_batch_size=env_config.num_envs,
        sample_sequence_length=horizon,
        period=1,
    )
    jitted_add = jax.jit(replay_buffer.add)
    jitted_sample = jax.jit(replay_buffer.sample)
    buffer_state = replay_buffer.init(dummy_transition)

    ##############################
    # Agent + encoder setup
    ##############################
    encoder = TrainState.create(
        apply_fn=encoder_module.apply,
        params=encoder_module.init(encoder_key, dummy_obs)['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(model_config.max_grad_norm),
            optax.adam(encoder_config.learning_rate),
        )
    )

    model = WorldModel.create(
        action_dim=np.prod(env.single_action_space.shape),
        encoder=encoder,
        **model_config,
        key=model_key
    )
    if model.action_dim >= 20:
        tdmpc_config.mppi_iterations += 2

    agent = TDMPC2.create(world_model=model, **tdmpc_config)
    global_step = 0

    ##############################
    # Checkpoint setup
    ##############################
    options = ocp.CheckpointManagerOptions(
        max_to_keep=1, save_interval_steps=cfg['save_interval_steps']
    )
    checkpoint_path = os.path.join(output_dir, 'checkpoint')
    with ocp.CheckpointManager(
        checkpoint_path,
        options=options,
        item_names=('agent', 'global_step', 'buffer_state')
    ) as mngr:
        if mngr.latest_step() is not None:
            print('Checkpoint folder found, restoring from', mngr.latest_step())
            abstract_buffer_state = jax.tree.map(
                ocp.utils.to_shape_dtype_struct, buffer_state
            )
            restored = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(
                    agent=ocp.args.StandardRestore(agent),
                    global_step=ocp.args.JsonRestore(),
                    buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
                )
            )
            agent, global_step = restored.agent, restored.global_step
            buffer_state = restored.buffer_state
        else:
            print('No checkpoint folder found, starting from scratch')
            mngr.save(
                global_step,
                args=ocp.args.Composite(
                    agent=ocp.args.StandardSave(agent),
                    global_step=ocp.args.JsonSave(global_step),
                    buffer_state=ocp.args.StandardSave(buffer_state),
                ),
            )
            mngr.wait_until_finished()

        ##############################
        # Training loop
        ##############################
        ep_count = np.zeros(env_config.num_envs, dtype=int)
        prev_logged_step = global_step
        plan = None
        observation, _ = env.reset(seed=cfg.seed)

        T = 500
        seed_steps = int(
            max(5 * T, 1000) * env_config.num_envs * env_config.utd_ratio
        )
        pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
        done = np.zeros(env_config.num_envs, dtype=bool)

        for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
            if global_step <= seed_steps:
                action = env.action_space.sample()
            else:
                rng, action_key = jax.random.split(rng)
                action, plan = agent.act(
                    observation,
                    prev_plan=plan,
                    mpc=True,
                    deterministic=False,
                    train=True,
                    key=action_key
                )
                action = np.array(action)

            next_observation, reward, terminated, truncated, info = env.step(action)

            # Build transition dict for flashbax — shape (num_envs, ...) per field
            # Only insert for envs that were not already done (same logic as base)

            transition = jtu.tree_map(
                lambda x: jnp.expand_dims(jnp.array(x), axis=1),  # (num_envs, 1, ...)
                {
                    'obs': observation,
                    'action': action,
                    'reward': reward.astype(np.float32),
                    'next_obs': next_observation,
                    'terminated': terminated.astype(np.float32),
                    'truncated': truncated.astype(np.float32),
                }
            )

            buffer_state = jitted_add(buffer_state, transition)
            observation = next_observation

            # Handle terminations/truncations
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                if plan is not None:
                    plan = (
                        plan[0].at[done].set(0),
                        plan[1].at[done].set(agent.max_plan_std)
                    )
                for ienv in range(env_config.num_envs):
                    if done[ienv]:
                        r = info['episode']['r'][ienv]
                        l = info['episode']['l'][ienv]
                        print(f"Episode {ep_count[ienv]}: r = {r:.2f}, l = {l}")
                        writer.scalar(f'episode/return', r, global_step + ienv)
                        writer.scalar(f'episode/length', l, global_step + ienv)
                        ep_count[ienv] += 1

            if global_step >= seed_steps:
                if global_step == seed_steps:
                    print('Pre-training on seed data...')
                    num_updates = seed_steps
                else:
                    num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

                rng, *update_keys = jax.random.split(rng, num_updates + 1)
                log_this_step = global_step >= prev_logged_step + cfg['log_interval_steps']
                if log_this_step:
                    all_train_info = defaultdict(list)
                    prev_logged_step = global_step

                for iupdate in range(num_updates):
                    # Sample from flashbax buffer
                    rng, sample_key = jax.random.split(rng)
                    # batch = replay_buffer.sample(buffer_state, sample_key)
                    batch = jitted_sample(buffer_state, sample_key)
                    # batch.experience shape: (batch_size, horizon+1, ...)
                    # swap to (horizon+1, batch_size, ...) to match agent.update expectation
                    experience = jtu.tree_map(
                        lambda x: jnp.swapaxes(x, 0, 1), batch.experience
                    )

                    agent, train_info = agent.update(
                        observations=experience['obs'],
                        actions=experience['action'],
                        rewards=experience['reward'],
                        next_observations=experience['next_obs'],
                        terminated=experience['terminated'],
                        truncated=experience['truncated'],
                        key=update_keys[iupdate]
                    )

                    if log_this_step:
                        for k, v in train_info.items():
                            all_train_info[k].append(np.array(v))

                if log_this_step:
                    for k, v in all_train_info.items():
                        writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
                        writer.scalar(f'train/{k}_std', np.std(v), global_step)

                mngr.save(
                    global_step,
                    args=ocp.args.Composite(
                        agent=ocp.args.StandardSave(agent),
                        global_step=ocp.args.JsonSave(global_step),
                        buffer_state=ocp.args.StandardSave(buffer_state),
                    ),
                )

            pbar.update(env_config.num_envs)
        pbar.close()


if __name__ == '__main__':
    train()