from continual_rl.policies.impala.impala_policy import ImpalaPolicy
from continual_rl.policies.mask.mask_policy_config import MaskPolicyConfig
from continual_rl.policies.mask.mask_monobeast import MaskMonobeast 

from continual_rl.policies.mask.nets_mask import ImpalaNetMask 


class MaskPolicy(ImpalaPolicy):
    """
    Implementation of Impala policy with Mask lifelong learning method.
    See link to original supermask approach in continual supervised learning:
    https://arxiv.org/abs/2006.14769
    """
    def __init__(self, config: MaskPolicyConfig, observation_space, action_spaces, impala_class: MaskMonobeast = None, policy_net_class=None):

        if impala_class is None:
            impala_class = MaskMonobeast
        if policy_net_class is None:
            policy_net_class = ImpalaNetMask

        super().__init__(config, observation_space, action_spaces, impala_class=impala_class,
                        policy_net_class=policy_net_class)

    #def get_environment_runner(self, task_spec):
    #    raise NotImplementedError

    #def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
    #    raise NotImplementedError

    #def train(self, storage_buffer):
    #    raise NotImplementedError

    #def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
    #    raise NotImplementedError

    #def load(self, output_path_dir):
    #    raise NotImplementedError
