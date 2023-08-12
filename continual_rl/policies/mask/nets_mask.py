import torch
import torch.nn as nn
import torch.nn.functional as F
from continual_rl.utils.utils import Utils

from continual_rl.policies.impala.nets import ImpalaNet
from continual_rl.policies.mask.mask_utils import MultitaskMaskLinear
from continual_rl.policies.mask.mask_utils import MultitaskMaskLinearSparse
from continual_rl.policies.mask.mask_utils import MultitaskMaskConv2d
from continual_rl.policies.mask.mask_utils import MultitaskMaskConv2dSparse
import continual_rl.policies.mask.mask_utils as mask_utils

class ImpalaNetMask(ImpalaNet):
    # overrides the parent init class with some key updates
    def __init__(self, observation_space, action_spaces, num_tasks, new_task_mask, use_lstm=False, \
        conv_net=None):
        super().__init__(observation_space, action_spaces, use_lstm, conv_net)
        self.use_lstm = use_lstm
        self.num_actions = Utils.get_max_discrete_action_space(action_spaces).n
        self._action_spaces = action_spaces  # The max number of actions - the policy's output size is always this
        self._current_action_size = None  # Set by the environment_runner
        self._observation_space = observation_space

        if conv_net is None:
            # The conv net gets channels and time merged together (mimicking the original FrameStacking)
            combined_observation_size = [observation_space.shape[0] * observation_space.shape[1],
                                         observation_space.shape[2],
                                         observation_space.shape[3]]
            self._conv_net = get_network_for_size(combined_observation_size, num_tasks,new_task_mask)
        else:
            raise NotImplementedError
            #self._conv_net = conv_net

        # set default task id to 0
        mask_utils.set_model_task(self._conv_net, 0)

        # FC output size + one-hot of last action + last reward.
        # NOTE update deviation from parent class init
        core_output_size = self._conv_net.output_size + self.num_actions + 1
        self.policy = MultitaskMaskLinear(core_output_size, self.num_actions, num_tasks=num_tasks, \
            new_mask_type=new_task_mask)
        self.baseline = MultitaskMaskLinear(core_output_size, 1, num_tasks=num_tasks, \
            new_mask_type=new_task_mask)
        # set default task id to 0
        mask_utils.set_model_task(self.policy, 0)
        mask_utils.set_model_task(self.baseline, 0)

        # used by update_running_moments()
        # second moment is variance
        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

    def forward(self, inputs, action_space_id, task_id, core_state=()):
        x = inputs["frame"]  # [T, B, S, C, H, W]. T=timesteps in collection, S=stacked frames
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = torch.flatten(x, 1, 2)  # Merge stacked frames and channels.
        x = x.float() / self._observation_space.high.max()
        x = self._conv_net(x, task_id) # NOTE
        x = F.relu(x)

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1).float()
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output, task_id) # NOTE
        baseline = self.baseline(core_output, task_id) # NOTE

        # Used to select the action appropriate for this task (might be from a reduced set)
        current_action_size = self._action_spaces[action_space_id].n
        policy_logits_subset = policy_logits[:, :current_action_size]

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits_subset, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits_subset, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

# Masked conv net utils. borrowed from: 
# ./continual_rl/continual_rl/utils/common_nets.py
def get_network_for_size(size, num_tasks, new_task_mask):
    """
    Size is expected to be [channel, dim, dim]
    """
    size = list(size)  # In case the input is a tuple
    if size[-2:] == [7, 7]:
        net = MaskedConvNet7x7
    elif size[-2:] == [28, 28]:
        net = MaskedConvNet28x28
    elif size[-2:] == [84, 84]:
        net = MaskedConvNet84x84
    elif size[-2:] == [64, 64]:
        # just use 84x84, it should compute output dim
        net = MaskedConvNet84x84
    else:
        raise AttributeError("Unexpected input size")

    return net(size, num_tasks, new_task_mask)


class ModelUtils(object):
    """
    Allows for images larger than their stated minimums, and will auto-compute the output size accordingly
    """
    @classmethod
    def compute_output_size(cls, net, observation_size):
        dummy_input = torch.zeros(observation_size).unsqueeze(0)  # Observation size doesn't include batch, so add it
        #dummy_output = net(dummy_input).squeeze(0)  # Remove batch
        dummy_output = net(dummy_input, task_id=0).squeeze(0)  # Remove batch. NOTE quick fix version
        output_size = dummy_output.shape[0]
        return output_size


class MaskedCommonConv(nn.Module):
    def __init__(self, conv_net, post_flatten, output_size, num_tasks, new_task_mask):
        super().__init__()
        self._conv_net = conv_net
        self._post_flatten = post_flatten
        self.output_size = output_size
        self.num_tasks = num_tasks
        self.new_task_mask = new_task_mask

    def forward(self, x, task_id):
        x = self._conv_net(x.float(), task_id)
        x = self._post_flatten(x, task_id)
        return x


class MaskSequentialModel(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, task_id):
        for module in self:
            if isinstance(module, MultitaskMaskConv2d) or isinstance(module, MultitaskMaskLinear):
                x = module(x, task_id)
            else:
                x = module(x)
        return x

class MaskedConvNet84x84(MaskedCommonConv):
    def __init__(self, observation_shape, num_tasks, new_task_mask):
        # This is the same as used in AtariNet in Impala (torchbeast implementation)
        output_size = 512
        in_channels = observation_shape[0]

        conv_net = MaskSequentialModel(
            MultitaskMaskConv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, \
                num_tasks=num_tasks, new_mask_type=new_task_mask),
            nn.ReLU(),
            MultitaskMaskConv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, \
                num_tasks=num_tasks, new_mask_type=new_task_mask),
            nn.ReLU(),
            MultitaskMaskConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, \
                num_tasks=num_tasks, new_mask_type=new_task_mask),
            nn.ReLU(),
            nn.Flatten())

        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = MultitaskMaskLinear(intermediate_dim, output_size, num_tasks=num_tasks, \
            new_mask_type=new_task_mask)
        super().__init__(conv_net, post_flatten, output_size, num_tasks, new_task_mask)


class MaskedConvNet28x28(MaskedCommonConv):
    def __init__(self, observation_shape, num_tasks, new_task_mask):
        output_size = 32
        conv_net = MaskSequentialModel(
            MultitaskMaskConv2d(observation_shape[0], 24, kernel_size=5, num_tasks=num_tasks, \
                new_mask_type=new_task_mask),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),  # TODO: this is new... (check)
            MultitaskMaskConv2d(24, 48, kernel_size=5, num_tasks=num_tasks, \
                new_mask_type=new_task_mask),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = MultitaskMaskLinear(intermediate_dim, output_size, num_tasks=num_tasks, \
            new_mask_type=new_task_mask)
        super().__init__(conv_net, post_flatten, output_size, num_tasks)


class MaskedConvNet7x7(MaskedCommonConv):
    def __init__(self, observation_shape, num_tasks, new_task_mask):
        output_size = 64
        conv_net = MaskSequentialModel(
            MultitaskMaskConv2d(observation_shape[0], 32, kernel_size=2, num_tasks=num_tasks, \
                new_mask_type=new_task_mask),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            MultitaskMaskConv2d(32, 64, kernel_size=2, num_tasks=num_tasks, \
                new_mask_type=new_task_mask),
            nn.ReLU(),
            MultitaskMaskConv2d(64, 128, kernel_size=2, num_tasks=num_tasks, \
                new_mask_type=new_task_mask),
            nn.ReLU(),
            nn.Flatten()
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = MultitaskMaskLinear(intermediate_dim, output_size, num_tasks=num_tasks, \
            new_mask_type=new_task_mask)
        super().__init__(conv_net, post_flatten, output_size, num_tasks)
