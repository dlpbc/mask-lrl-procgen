from continual_rl.policies.config_base import ConfigBase
from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig

import continual_rl.policies.mask.mask_utils as mask_utils

class MaskPolicyConfig(ImpalaPolicyConfig):

    def __init__(self):
        super().__init__()
        self.batch_size = 20
        # following parameters specified in EWCPolicyConfig
        self.unroll_length = 20
        self.epsilon = 0.1  # RMSProp epsilon
        self.learning_rate = 0.0006
        self.entropy_cost = 0.01
        self.reward_clipping = "abs_one"
        self.baseline_cost = 0.5
        self.discounting = 0.99

        self.large_file_path = None  # No default, since it can be very large and we want no surprises
        self.num_tasks = None # to be overwritten by experiment config
        self.new_task_mask = mask_utils.NEW_MASK_RANDOM

    def _load_from_dict_internal(self, config_dict):
        self._auto_load_class_parameters(config_dict)
        return self
