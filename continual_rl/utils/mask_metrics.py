import argparse
import numpy as np
from continual_rl.utils.metrics import Metrics

# see https://github.com/plotly/Kaleido/issues/101
import plotly.io as pio
pio.kaleido.scope.mathjax = None  # Prevents a weird "Loading MathJax" artifact in rendering the pdf


#TASKS_ATARI = {
#    "0-SpaceInvaders": dict(i=0, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[0, 50e6], [300e6, 350e6]], showlegend=False),
#    "1-Krull": dict(i=1, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[50e6, 100e6], [350e6, 400e6]], showlegend=False),
#    "2-BeamRider": dict(i=2, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[100e6, 150e6], [400e6, 450e6]], showlegend=True),
#    "3-Hero": dict(i=3, y_range=[0, 5e4], yaxis_dtick=1e4, train_regions=[[150e6, 200e6], [450e6, 500e6]], showlegend=False),
#    "4-StarGunner": dict(i=4, y_range=[0, 10e4], yaxis_dtick=2e4, train_regions=[[200e6, 250e6], [500e6, 550e6]], showlegend=False),
#    "5-MsPacman": dict(i=5, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[250e6, 300e6], [550e6, 600e6]], showlegend=True),
#}
TASKS_ATARI = {
    "0-SpaceInvaders": dict(i=0, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[0, 50e6],], showlegend=False),
    "1-Krull": dict(i=1, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[50e6, 100e6],], showlegend=False),
    "2-BeamRider": dict(i=2, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[100e6, 150e6],], showlegend=True),
    "3-Hero": dict(i=3, y_range=[0, 5e4], yaxis_dtick=1e4, train_regions=[[150e6, 200e6],], showlegend=False),
    "4-StarGunner": dict(i=4, y_range=[0, 10e4], yaxis_dtick=2e4, train_regions=[[200e6, 250e6],], showlegend=False),
    "5-MsPacman": dict(i=5, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[250e6, 300e6],], showlegend=True),
}


MODELS_ATARI = {
    "IMPALA": dict(
        name='impala',
        runs=[f'impala_atari/0', f'impala_atari/0'],
        color='rgba(77, 102, 133, 1)',
        color_alpha=0.2,
    ),
    #"MASK RI": dict(
    #    name='mask random init',
    #    runs=[f'mask_atari_random_init/0', f'mask_atari_random_init/0'],
    #    color='rgba(0, 48, 255, 1)',
    #    color_alpha=0.2,
    #),
    "MASK LC": dict(
        name='mask linear comb',
        runs=[f'mask_atari_linear_comb/0', f'mask_atari_linear_comb/0',],
        color='rgba(255, 0, 154, 1)',
        color_alpha=0.2,
    ),
}
ATARI = dict(
    models=MODELS_ATARI,
    tasks=TASKS_ATARI,
    #num_cycles=2,
    num_cycles=1, # NOTE
    num_cycles_for_forgetting=1,
    num_task_steps=50e6,
    grid_size=[2, 3],
    which_exp='atari',
    rolling_mean_count=20,
    filter='ma',
    #xaxis_tickvals=list(np.arange(0, 600e6 + 1, 300e6)),
    xaxis_tickvals=list(np.arange(0, 300e6 + 1, 300e6)), # NOTE
    cache_dir='tmp' #'tmp/cache/data_pkls/atari/',
)


TASKS_PROCGEN = {
    "0-Climber": dict(i=0, eval_i=1, y_range=[0., 1.75], yaxis_dtick=0.25, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(0, 6 * 5, 6)]),
    "1-Dodgeball": dict(i=2, eval_i=3, y_range=[0., 2.5], yaxis_dtick=0.5, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(1, 6 * 5, 6)]),
    "2-Ninja": dict(i=4, eval_i=5, y_range=[0., 5.], yaxis_dtick=1.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(2, 6 * 5, 6)]),
    "3-Starpilot": dict(i=6, eval_i=7, y_range=[0., 10.], yaxis_dtick=2.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(3, 6 * 5, 6)]),
    "4-Bigfish": dict(i=8, eval_i=9, y_range=[0., 10.], yaxis_dtick=2.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(4, 6 * 5, 6)]),
    "5-Fruitbot": dict(i=10, eval_i=11, y_range=[-4, 3], yaxis_dtick=1, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(5, 6 * 5, 6)]),
}
MODELS_PROCGEN = {
    "IMPALA": dict(
        name='impala',
        runs=[f'run_{i}/impala_procgen/0' for i in range(1, 4)],
        color='rgba(77, 102, 133, 1)',
        color_alpha=0.2,
    ),
    "MASK RI": dict(
        name='mask random init',
        runs=[f'run_{i}/mask_procgen_random_init/0' for i in range(1, 4)],
        color='rgba(0, 48, 255, 1)',
        color_alpha=0.2,
    ),
    "MASK LC": dict(
        name='mask linear comb',
        runs=[f'run_{i}/mask_procgen_linear_comb/0' for i in range(1, 4)],
        color='rgba(255, 0, 154, 1)',
        color_alpha=0.2,
    ),
    "MASK BLC": dict(
        name='mask linear comb',
        runs=[f'run_{i}/mask_procgen_linear_comb_blc/0' for i in range(1, 4)],
        color='rgba(255, 165, 0, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'run_{i}/clear_procgen/0' for i in range(1, 4)],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
    "ONLINE EWC": dict(
        name='online ewc',
        runs=[f'run_{i}/online_ewc_procgen/0' for i in range(1, 4)],
        color='rgba(106, 166, 110, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'run_{i}/pnc_procgen/0' for i in range(1, 4)],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
}
PROCGEN = dict(
    models=MODELS_PROCGEN,
    tasks=TASKS_PROCGEN,
    rolling_mean_count=20,
    filter='ma',
    num_cycles=5,
    num_cycles_for_forgetting=1,
    num_task_steps=5e6,
    grid_size=[2, 3],
    which_exp='procgen',
    xaxis_tickvals=list(np.arange(0, 150e6 + 1, 30e6)),
    cache_dir='tmp' #/cache/data_pkls/procgen_resblocks/',
)

TO_PLOT = dict(
    tag_base='eval_reward',
    cache_dir='tmp/',
    legend_size=30,
    title_size=40,
    axis_size=20,
    axis_label_size=30,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='experiment dir')
    args = parser.parse_args()
    TO_PLOT['exp_dir'] = args.d

    #exp_data = ATARI
    exp_data = PROCGEN
    TO_PLOT.update(**exp_data)

    metrics = Metrics(TO_PLOT)
    metrics.visualize()
