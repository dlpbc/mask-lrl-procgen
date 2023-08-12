from continual_rl.policies.impala.impala_policy import ImpalaPolicy
from continual_rl.policies.mask.mask_policy_config import MaskPolicyConfig
from continual_rl.policies.mask.mask_monobeast import MaskMonobeast 

from continual_rl.policies.mask.nets_mask import ImpalaNetMask 

import continual_rl.policies.mask.mask_utils as mask_utils


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

        self.trained_tasks = []
        self.new_task = False
        self.train_task_id = None
        self.eval_task_id = None

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


    def task_train_start(self, task_id):
        self.train_task_id = task_id
        if task_id not in self.trained_tasks:
            self.new_task = True
        mask_utils.set_model_task(self.impala_trainer.actor_model, task_id)
        mask_utils.set_model_task(self.impala_trainer.learner_model, task_id)
        return

    def task_train_end(self):
        if self.new_task:
            # order: consolidate -> cache -> increment number of learnt task
            # consolidate mask
            mask_utils.consolidate_mask(self.impala_trainer.actor_model)
            mask_utils.consolidate_mask(self.impala_trainer.learner_model)
            # cache mask
            mask_utils.cache_masks(self.impala_trainer.actor_model)
            mask_utils.cache_masks(self.impala_trainer.learner_model)
            # number of learnt tasks
            self.trained_tasks.append(self.train_task_id)
            num_tasks = len(self.trained_tasks)
            mask_utils.set_num_tasks_learned(self.impala_trainer.actor_model, num_tasks)
            mask_utils.set_num_tasks_learned(self.impala_trainer.learner_model, num_tasks)
            self.new_task = False
        else:
            # cache mask
            mask_utils.cache_masks(self.impala_trainer.actor_model)
            mask_utils.cache_masks(self.impala_trainer.learner_model)
        self.train_task_id = None
        return

    def task_eval_start(self, task_id):
        self.eval_task_id = task_id
        mask_utils.set_model_task(self.impala_trainer.actor_model, task_id)
        mask_utils.set_model_task(self.impala_trainer.learner_model, task_id)
        return

    def task_eval_end(self):
        self.eval_task_id = None
        if self.train_task_id is not None:
            mask_utils.set_model_task(self.impala_trainer.actor_model, self.train_task_id)
            mask_utils.set_model_task(self.impala_trainer.learner_model, self.train_task_id)
        return
