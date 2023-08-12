import os
import logging
import pprint
import time
import timeit
import traceback
import typing
import copy
import psutil
import numpy as np
import queue
import cloudpickle
from torch.multiprocessing import Pool
import threading
import json
import shutil
import signal

import torch
import multiprocessing as py_mp
from torch import multiprocessing as mp

from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers
from continual_rl.policies.impala.torchbeast.core import environment
from continual_rl.policies.impala.torchbeast.core import prof
from continual_rl.policies.impala.torchbeast.core import vtrace
from continual_rl.utils.utils import Utils

class MaskMonobeast(Monobeast):
    """
    An implementation of Impala + Mask.
    See link to original supermask approach in continual supervised learning:
    https://arxiv.org/abs/2006.14769
    """

    def __init__(self, model_flags, observation_space, action_spaces, policy_class):
        super().__init__(model_flags, observation_space, action_spaces, policy_class)

    # Core Monobeast functionality
    def setup(self, model_flags, observation_space, action_spaces, policy_class):
        os.environ["OMP_NUM_THREADS"] = "1"
        logging.basicConfig(
            format=(
                "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
            ),
            level=0,
        )

        logger = Utils.create_logger(os.path.join(model_flags.savedir, "impala_logs.log"))
        plogger = Utils.create_logger(os.path.join(model_flags.savedir, "impala_results.log"))

        checkpointpath = os.path.join(model_flags.savedir, "model.tar")

        if model_flags.num_buffers is None:  # Set sensible default for num_buffers.
            model_flags.num_buffers = max(2 * model_flags.num_actors, model_flags.batch_size)
        if model_flags.num_actors >= model_flags.num_buffers:
            raise ValueError("num_buffers should be larger than num_actors")
        if model_flags.num_buffers < model_flags.batch_size:
            raise ValueError("num_buffers should be larger than batch_size")

        # Convert the device string into an actual device
        model_flags.device = torch.device(model_flags.device)

        # NOTE updated in comparison to parent method
        model = policy_class(observation_space, action_spaces, model_flags.num_tasks, \
            model_flags.new_task_mask, model_flags.use_lstm) 
        buffers = self.create_buffers(model_flags, observation_space.shape, model.num_actions)

        model.share_memory()

        # NOTE updated in comparison to parent method
        learner_model = policy_class(
            observation_space, action_spaces, model_flags.num_tasks, model_flags.new_task_mask, \
            model_flags.use_lstm
        ).to(device=model_flags.device)

        if model_flags.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(
                learner_model.parameters(),
                lr=model_flags.learning_rate,
                momentum=model_flags.momentum,
                eps=model_flags.epsilon,
                alpha=model_flags.alpha,
            )
        elif model_flags.optimizer == "adam":
            optimizer = torch.optim.Adam(
                learner_model.parameters(),
                lr=model_flags.learning_rate,
            )
        else:
            raise ValueError(f"Unsupported optimizer type {model_flags.optimizer}.")

        return buffers, model, learner_model, optimizer, plogger, logger, checkpointpath

    def act(
            self,
            model_flags,
            task_flags,
            actor_index: int,
            free_queue: py_mp.Queue,
            full_queue: py_mp.Queue,
            model: torch.nn.Module,
            buffers: Buffers,
            initial_agent_state_buffers,
    ):
        env = None
        try:
            self.logger.info("Actor %i started.", actor_index)
            timings = prof.Timings()  # Keep track of how fast things are.

            gym_env, seed = Utils.make_env(task_flags.env_spec, create_seed=True)
            self.logger.info(f"Environment and libraries setup with seed {seed}")

            # Parameters involved in rendering behavior video
            observations_to_render = []  # Only populated by actor 0

            env = environment.Environment(gym_env)
            env_output = env.initial()
            agent_state = model.initial_state(batch_size=1)
            # NOTE line updated in comparison with parent implementation
            # example task_flag.task_id: atari_6_tasks_2_cycles_0
            #int_task_id = int(task_flags.task_id.split('_')[-1])
            try:
                int_task_id = int(task_flags.task_id.split('_')[-1])
            except:
                int_task_id = int(task_flags.task_id.split('_')[-2])
            agent_output, unused_state = model(env_output, task_flags.action_space_id, int_task_id, agent_state)

            # Make sure to kill the env cleanly if a terminate signal is passed. (Will not go through the finally)
            def end_task(*args):
                env.close()

            signal.signal(signal.SIGTERM, end_task)

            while True:
                index = free_queue.get()
                if index is None:
                    break

                # Write old rollout end.
                for key in env_output:
                    buffers[key][index][0, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][0, ...] = agent_output[key]
                for i, tensor in enumerate(agent_state):
                    initial_agent_state_buffers[index][i][...] = tensor

                # Do new rollout.
                for t in range(model_flags.unroll_length):
                    timings.reset()

                    with torch.no_grad():
                        # NOTE line updated in comparison with parent implementation
                        # example task_flag.task_id: atari_6_tasks_2_cycles_0
                        #int_task_id = int(task_flags.task_id.split('_')[-1])
                        try:
                            int_task_id = int(task_flags.task_id.split('_')[-1])
                        except:
                            int_task_id = int(task_flags.task_id.split('_')[-2])
                        agent_output, agent_state = model(env_output, task_flags.action_space_id, int_task_id, agent_state)

                    timings.time("model")

                    env_output = env.step(agent_output["action"])

                    timings.time("step")

                    for key in env_output:
                        buffers[key][index][t + 1, ...] = env_output[key]
                    for key in agent_output:
                        buffers[key][index][t + 1, ...] = agent_output[key]

                    # Save off video if appropriate
                    if actor_index == 0:
                        if env_output['done'].squeeze():
                            # If we have a video in there, replace it with this new one
                            try:
                                self._videos_to_log.get(timeout=1)
                            except queue.Empty:
                                pass
                            except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError, RuntimeError) as e:
                                # Sometimes it seems like the videos_to_log socket fails. Since video logging is not
                                # mission-critical, just let it go.
                                self.logger.warning(
                                    f"Video logging socket seems to have failed with error {e}. Aborting video log.")
                                pass

                            self._videos_to_log.put(copy.deepcopy(observations_to_render))
                            observations_to_render.clear()

                        observations_to_render.append(env_output['frame'].squeeze(0).squeeze(0)[-1])

                    timings.time("write")

                new_buffers = {key: buffers[key][index] for key in buffers.keys()}
                self.on_act_unroll_complete(task_flags, actor_index, agent_output, env_output, new_buffers)
                full_queue.put(index)

            if actor_index == 0:
                self.logger.info("Actor %i: %s", actor_index, timings.summary())

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            self.logger.error(f"Exception in worker process {actor_index}: {e}")
            traceback.print_exc()
            print()
            raise e
        finally:
            self.logger.info(f"Finalizing actor {actor_index}")
            if env is not None:
                env.close()

    def compute_loss(self, model_flags, task_flags, learner_model, batch, initial_agent_state, with_custom_loss=True):
        # Note the action_space_id isn't really used - it's used to generate an action, but we use the action that
        # was already computed and executed

        # NOTE line updated in comparison with parent implementation
        # example task_flag.task_id: atari_6_tasks_2_cycles_0
        #int_task_id = int(task_flags.task_id.split('_')[-1])
        try:
            int_task_id = int(task_flags.task_id.split('_')[-1])
        except:
            int_task_id = int(task_flags.task_id.split('_')[-2])
        learner_outputs, unused_state = learner_model(batch, task_flags.action_space_id, int_task_id, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]

        # from https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/polybeast/polybeast_learner.py#L243
        if model_flags.normalize_reward:
            learner_model.update_running_moments(rewards)
            rewards /= learner_model.get_running_std()

        if model_flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif model_flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * model_flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = self.compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = model_flags.baseline_cost * self.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = model_flags.entropy_cost * self.compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss
        stats = {
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        if with_custom_loss: # auxilary terms for continual learning
            custom_loss, custom_stats = self.custom_loss(task_flags, learner_model, initial_agent_state)
            total_loss += custom_loss
            stats.update(custom_stats)

        return total_loss, stats, pg_loss, baseline_loss

    def custom_loss(self, task_flags, model, initial_agent_state):
        """
        Create a new loss. This is added to the existing losses before backprop. Any returned stats will be added
        to the logged stats. If a stat's key ends in "_loss", it'll automatically be plotted as well.
        This is run in each learner thread.
        :return: (loss, dict of stats)
        """
        return 0, {}

    #def train(self, task_flags):  # pylint: disable=too-many-branches, too-many-statements
    #    return super().train(task_flags)

    #def test(self, task_flags, num_episodes: int = 10):
    #    return super().test(task_flags, num_episodes)

    @staticmethod
    def _collect_test_episode(pickled_args):
        task_flags, logger, model = cloudpickle.loads(pickled_args)

        gym_env, seed = Utils.make_env(task_flags.env_spec, create_seed=True)
        logger.info(f"Environment and libraries setup with seed {seed}")
        env = environment.Environment(gym_env)
        observation = env.initial()
        done = False
        step = 0
        returns = []

        while not done:
            if task_flags.mode == "test_render":
                env.gym_env.render()

            # NOTE line updated in comparison with parent implementation
            # example task_flag.task_id: atari_6_tasks_2_cycles_0
            try:
                int_task_id = int(task_flags.task_id.split('_')[-1])
            except:
                int_task_id = int(task_flags.task_id.split('_')[-2])
            agent_outputs = model(observation, task_flags.action_space_id, int_task_id)

            policy_outputs, _ = agent_outputs
            observation = env.step(policy_outputs["action"])
            step += 1
            done = observation["done"].item() and not torch.isnan(observation["episode_return"])

            # NaN if the done was "fake" (e.g. Atari). We want real scores here so wait for the real return.
            if done:
                returns.append(observation["episode_return"].item())
                logger.info(
                    "Episode ended after %d steps. Return: %.1f",
                    observation["episode_step"].item(),
                    observation["episode_return"].item(),
                )

        env.close()
        return step, returns
