from continual_rl.policies.timestep_data_base import TimestepDataBase


class MaskTimestepData(TimestepDataBase):
    def __init__(self):
        super().__init__()
