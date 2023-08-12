import numpy as np
import torch
import torch.nn as nn
import threading
import os
from torch.nn import functional as F
import queue
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers
from continual_rl.utils.utils import Utils


class ClearReplayHandler(nn.Module):
    """
    Creates a general CLEAR replay buffer, applicable to any policy (currently based on Monobeast).

    An implementation of Experience Replay for Continual Learning (Rolnick et al, 2019):
    https://arxiv.org/pdf/1811.11682.pdf
    """
    # TODO: clearer API spec
    def __init__(self, policy, model_flags, observation_space, action_spaces):
        super().__init__()
        self._model_flags = model_flags
        self._policy = policy
        common_action_space = Utils.get_max_discrete_action_space(action_spaces)

        torch.multiprocessing.set_sharing_strategy(model_flags.torch_multiprocessing_sharing_strategy)

        # LSTMs not supported largely because they have not been validated; nothing extra is stored for them.
        assert not self._model_flags.use_lstm, "CLEAR does not presently support using LSTMs."
        assert not self._model_flags.one_rollout_per_actor or self._model_flags.num_actors >= int(self._model_flags.batch_size * self._model_flags.batch_replay_ratio), \
            "Each actor only gets sampled from once during training, so we need at least as many actors as batch_size"
        assert self._model_flags.large_file_path is not None, "Large file path must be specified"

        # We want the replay buffers to be created in the large_file_path,
        # but in a place characteristic to this experiment.
        # Be careful if the output_dir specified is very nested
        # (ie. Windows has max path length of 260 characters)
        # Could hash output_dir_str if this is a problem.
        output_dir_str = os.path.normpath(model_flags.output_dir).replace(os.path.sep, '-')
        permanent_path = os.path.join(
            model_flags.large_file_path,
            "file_backed",
            output_dir_str,
        )
        buffers_existed = os.path.exists(permanent_path)
        os.makedirs(permanent_path, exist_ok=True)

        self._entries_per_buffer = int(
            model_flags.replay_buffer_frames // (model_flags.unroll_length * model_flags.num_actors)
        )
        self._replay_buffers, self._temp_files = self._create_replay_buffers(
            model_flags,
            observation_space,
            common_action_space.n,
            self._entries_per_buffer,
            permanent_path,
            buffers_existed,
        )
        #self._replay_lock = threading.Lock()

        # Each replay batch needs to also have cloning losses applied to it
        # Keep track of them as they're generated, to ensure we apply losses to all. This doesn't currently
        # guarantee order - i.e. one learner thread might get one replay batch for training and a different for cloning
        #self._replay_batches_for_loss = queue.Queue()

    def _create_replay_buffers(
        self,
        model_flags,
        obs_space,
        num_actions,
        entries_per_buffer,
        permanent_path,
        buffers_existed,
    ):
        """
        Key differences from normal buffers:
        1. File-backed, so we can store more at a time
        2. Structured so that there are num_actors buffers, each with entries_per_buffer entries

        Each buffer entry has unroll_length size, so the number of frames stored is (roughly, because of integer
        rounding): num_actors * entries_per_buffer * unroll_length
        """
        # Get the standard specs, and also add the CLEAR-specific reservoir value
        specs = self._policy.create_buffer_specs(model_flags.unroll_length, obs_space, num_actions)
        # Note: one reservoir value per row
        specs["reservoir_val"] = dict(size=(1,), dtype=torch.float32)
        buffers: Buffers = {key: [] for key in specs}

        # Hold on to the file handle so it does not get deleted. Technically optional, as at least linux will
        # keep the file open even after deletion, but this way it is still visible in the location it was created
        temp_files = []

        for actor_id in range(model_flags.num_actors):
            for key in buffers:
                shape = (entries_per_buffer, *specs[key]["size"])
                permanent_file_name = f"replay_{actor_id}_{key}.fbt"
                new_tensor, temp_file = Utils.create_file_backed_tensor(
                    permanent_path,
                    shape,
                    specs[key]["dtype"],
                    permanent_file_name=permanent_file_name,
                )

                # reservoir_val needs to be 0'd out so we can use it to see if a row is filled
                # but this operation is slow, so leave the rest as-is
                # Only do this if we created the buffers anew
                if not buffers_existed and key == "reservoir_val":
                    new_tensor.zero_()

                buffers[key].append(new_tensor.share_memory_())
                temp_files.append(temp_file)

        return buffers, temp_files

    def _get_replay_buffer_filled_indices(self, replay_buffers, actor_index):
        """
        Get the indices in the replay buffer corresponding to the actor_index.
        """
        # We know that the reservoir value > 0 if it's been filled, so check for entries where it == 0
        buffer_indicator = replay_buffers['reservoir_val'][actor_index].squeeze(1)
        replay_indices = np.where(buffer_indicator != 0)[0]
        return replay_indices

    def _get_actor_unfilled_indices(self, actor_index, entries_per_buffer):
        """
        Get the unfilled entries in the actor's subset of the replay buffer using a set difference.
        """
        filled_indices = set(
            self._get_replay_buffer_filled_indices(self._replay_buffers, actor_index)
        )
        actor_id_set = set(range(0, entries_per_buffer))
        unfilled_indices = actor_id_set - filled_indices
        return unfilled_indices

    def _compute_policy_cloning_loss(self, old_logits, curr_logits):
        # KLDiv requires inputs to be log-probs, and targets to be probs
        old_policy = F.softmax(old_logits, dim=-1)
        curr_log_policy = F.log_softmax(curr_logits, dim=-1)
        kl_loss = torch.nn.KLDivLoss(reduction='sum')(curr_log_policy, old_policy.detach())
        return kl_loss

    def _compute_value_cloning_loss(self, old_value, curr_value):
        return torch.sum((curr_value - old_value.detach()) ** 2)

    def on_act_unroll_complete(self, task_flags, actor_index, new_buffers):
        """
        Every step, update the replay buffer using reservoir sampling.
        """
        # Compute a reservoir_val for the new entry, then, if the buffer is filled, throw out the entry with the lowest
        # reservoir_val and replace it with the new one. If the buffer it not filled, simply put it in the next spot
        # Using a new RandomState() because using np.random directly is not thread-safe
        random_state = np.random.RandomState()

        # > 0 so we can use reservoir_val==0 to indicate unfilled
        new_entry_reservoir_val = random_state.uniform(0.001, 1.0)
        to_populate_replay_index = None
        unfilled_indices = self._get_actor_unfilled_indices(actor_index, self._entries_per_buffer)

        actor_replay_reservoir_vals = self._replay_buffers['reservoir_val'][actor_index]

        if len(unfilled_indices) > 0:
            current_replay_index = min(unfilled_indices)
            to_populate_replay_index = current_replay_index
        else:
            # If we've filled our quota, we need to find something to throw out.
            reservoir_threshold = actor_replay_reservoir_vals.min()

            # If our new value is higher than our existing minimum, replace that one with this new data
            if new_entry_reservoir_val > reservoir_threshold:
                to_populate_replay_index = np.argmin(actor_replay_reservoir_vals)

        # Do the replacement into the buffer, and update the reservoir_vals list
        if to_populate_replay_index is not None:
            #with self._replay_lock:
            # TODO: passing locks to processes is problematic, so ...testin
            actor_replay_reservoir_vals[to_populate_replay_index][0] = new_entry_reservoir_val
            for key in new_buffers.keys():
                if key == 'reservoir_val':
                    continue
                if key in self._replay_buffers:
                    self._replay_buffers[key][actor_index][to_populate_replay_index][...] = new_buffers[key]

    def get_batch_for_training(self, batch, initial_agent_state):
        """
        Augment the batch with entries from our replay buffer.
        """
        # Select a random batch set of replay buffers to add also. Only select from ones that have been filled
        shuffled_subset = []  # Will contain a list of tuples of (actor_index, buffer_index)

        # We only allow each actor to be sampled from once, to reduce variance, and for parity with the original
        # paper
        actor_indices = list(range(self._model_flags.num_actors))
        replay_entry_count = int(self._model_flags.batch_size * self._model_flags.batch_replay_ratio)
        assert replay_entry_count > 0, "Attempting to run CLEAR without actually using any replay buffer entries."

        random_state = np.random.RandomState()

        #with self._replay_lock:
        # Select a random actor, and from that, a random buffer entry.
        for _ in range(replay_entry_count):
            # Pick an actor and remove it from our options
            actor_index = random_state.choice(actor_indices)

            # If we are only taking one rollout per actor, remove the actor from the running
            # TODO: in not-one-rollout-mode we can technically select the same batch multiple times right now
            if self._model_flags.one_rollout_per_actor:
                actor_indices.remove(actor_index)

            # From that actor's set of available indices, pick one randomly.
            replay_indices = self._get_replay_buffer_filled_indices(self._replay_buffers, actor_index=actor_index)
            if len(replay_indices) > 0:
                buffer_index = random_state.choice(replay_indices)
                shuffled_subset.append((actor_index, buffer_index))

        if len(shuffled_subset) > 0:
            replay_batch = {
                # Get the actor_index and entry_id from the raw id
                key: torch.stack([self._replay_buffers[key][actor_id][buffer_id]
                                    for actor_id, buffer_id in shuffled_subset], dim=1)
                for key in self._replay_buffers
            }

            replay_entries_retrieved = torch.sum(replay_batch["reservoir_val"] > 0)
            assert replay_entries_retrieved <= replay_entry_count, \
                f"Incorrect replay entries retrieved. Expected at most {replay_entry_count} got {replay_entries_retrieved}"

            replay_batch = {
                k: t.to(device=self._model_flags.device, non_blocking=True)
                for k, t in replay_batch.items()
            }

            # Combine the replay in with the recent entries
            combo_batch = {
                key: torch.cat((batch[key], replay_batch[key]), dim=1) for key in batch if key in replay_batch
            }

            # Augment the initial agent state for LSTM, so the size matches
            # TODO: do I need to save off and pipe through the OG batch initial states...?
            replay_initial_states = self._policy.initial_state(batch_size=replay_entries_retrieved) 
            combo_initial_states = []
            for state_id, state in enumerate(initial_agent_state):
                state = torch.cat((state, replay_initial_states[state_id].to(device=self._model_flags.device)), dim=1)
                combo_initial_states.append(state)
            initial_agent_state = tuple(combo_initial_states)

            # Store the batch so we can generate some losses with it
            #self._replay_batches_for_loss.put((combo_batch, combo_initial_states))

        else:
            combo_batch = batch

        return combo_batch, initial_agent_state

    def custom_loss(self, task_flags, model, initial_agent_state, batch):  # TODO: clean up initial state + batch. Right now batch also has initial state, for hackrl at least
        """
        Compute the policy and value cloning losses
        """
        # If the get doesn't happen basically immediately, it's not happening
        replay_batch = batch #self._replay_batches_for_loss.get(timeout=5) -- TODO why did I do this queue thing? seemed necessary at the time, seems...not now
        combo_agent_state = initial_agent_state  # TODO: renames just being lazy

        # TODO: again...very hacky, definitely not the right way to expose this... Probably another policy function
        if "action_space_id" in task_flags:
            replay_learner_outputs, unused_state = model(replay_batch, task_flags.action_space_id, combo_agent_state)
        else:
            replay_learner_outputs, unused_state = model(replay_batch, combo_agent_state)

        replay_batch_policy = replay_batch['policy_logits']
        current_policy = replay_learner_outputs['policy_logits']
        policy_cloning_loss = self._model_flags.policy_cloning_cost * self._compute_policy_cloning_loss(replay_batch_policy, current_policy)

        replay_batch_baseline = replay_batch['baseline']
        current_baseline = replay_learner_outputs['baseline']
        value_cloning_loss = self._model_flags.value_cloning_cost * self._compute_value_cloning_loss(replay_batch_baseline, current_baseline)

        cloning_loss = policy_cloning_loss + value_cloning_loss
        stats = {
            "policy_cloning_loss": policy_cloning_loss.item(),
            "value_cloning_loss": value_cloning_loss.item(),
        }

        return cloning_loss, stats


class ClearMonobeast(Monobeast):
    def __init__(self, model_flags, observation_space, action_spaces, policy_class):
        super().__init__(model_flags, observation_space, action_spaces, policy_class)
        self._clear_wrapper = ClearReplayHandler(self, model_flags, observation_space, action_spaces)

    def initial_state(self, batch_size):
        # TODO: doesn't exactly fit here but...going with it for now
        return self.actor_model.initial_state(batch_size)
        
    def on_act_unroll_complete(self, task_flags, actor_index, new_buffers):
        return self._clear_wrapper.on_act_unroll_complete(task_flags, actor_index, new_buffers)

    def get_batch_for_training(self, batch, initial_agent_state):
        return self._clear_wrapper.get_batch_for_training(batch, initial_agent_state)

    def custom_loss(self, task_flags, model, initial_agent_state, batch):
        return self._clear_wrapper.custom_loss(task_flags, model, initial_agent_state, batch)
