import numpy as np
import pandas as pd
import ijson
from decimal import Decimal
from sys import stderr
from os import path


def fancyprint(msg: str, dim: bool, bold: bool, type: str, color: str):
    styles = ''
    if bold:
        styles += '\033[1m'
    if dim:
        styles += '\033[2m'
    print(
        f'\033[1;{color}m{type} \033[0m{styles} {msg}\033[0m', file=stderr)


def err(msg: str, dim: bool = False, bold: bool = False):
    fancyprint(msg, dim, bold, 'STRAGGLER ERR: ', "31")


def warn(msg: str, dim: bool = False, bold: bool = False):
    fancyprint(msg, dim, bold, 'STRAGGLER WARN:', "33")


def info(msg: str, dim: bool = False, bold: bool = False):
    fancyprint(msg, dim, bold, 'STRAGGLER INFO:', "34")


def dtoi(ts):
    return np.int64(ts * Decimal('1000'))


def kernel_link(line, data):
    data.append({
        'name': line['name'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'stream': line['args']['stream'],
    })


def json_to_pandas(json_fn: str) -> pd.DataFrame:
    kernel_data = []

    with open(json_fn, 'r') as fp:
        info(f'Reading {path.basename(json_fn)}...')
        for line in ijson.items(fp, 'traceEvents.item'):
            if 'cat' in line and line['cat'] == 'kernel':
                kernel_link(
                    line,
                    kernel_data,
                )
            else:
                continue

    return pd.DataFrame(kernel_data)


def merge_traces(pytorch_traces):
    gpu_map = {i: trace_fn for i, trace_fn in enumerate(pytorch_traces)}
    for gpu, trace_fn in gpu_map.items():
        info(f'Mapping {path.basename(trace_fn)} to GPU{gpu}')
    return pd.concat(
        (
            df.assign(gpu=i)
            for i, df in
            enumerate(map(json_to_pandas, pytorch_traces))
        ),
        ignore_index=True
    )


def leader_value(df, agg=True, use_last=False, use_max=False, use_sum=False):
    df_compute = df[~df['name'].str.startswith('ncclDevKernel')].copy()
    df_compute['_ki'] = (
        df_compute
        .groupby(['gpu', 'name'])
        .cumcount()
    )
    straggler = (
        df_compute
        .groupby(['name', '_ki'])
        ['ts']
        .max()
        .rename('straggler_ts')
    )
    df_ = (
        df_compute
        .merge(
            straggler,
            on=['name', '_ki'],
            how='left'
        )
    )
    df_['lead'] = df_['straggler_ts'] - df_['ts']
    df_.drop(columns='straggler_ts', inplace=True)
    df_.sort_values(['gpu', 'ts'], inplace=True)
    if agg:
        if use_sum:
            return (
                df_
                .groupby('gpu')['lead']
                .sum()
                .reset_index()
            )
        elif use_max:
            return (
                df_
                .groupby('gpu')['lead']
                .max()
                .reset_index()
            )
        elif use_last:
            return (
                df_
                .groupby('gpu')['lead']
                .last()
                .reset_index()
            )
        else:
            raise ValueError("One needs to be true")
    else:
        return df_[['gpu', 'lead']]


def no_overlap(df):
    df = df.copy()
    df['end'] = df['ts'] + df['dur']

    df_compute = df[~df['name'].str.startswith('ncclDevKernel')].copy()
    df_compute['_ki'] = (
        df_compute
        .groupby(['gpu', 'name'])
        .cumcount()
    )

    df_overlap = df[df['name'].str.startswith('ncclDevKernel')].copy()

    no_overlap = pd.Series(True, index=df_compute.index)
    for _, row_overlap in df_overlap.iterrows():
        overlap_condition = (df_compute['ts'] < row_overlap['end']) & (
            df_compute['end'] > row_overlap['ts'])
        no_overlap &= ~overlap_condition
    df_compute = df_compute[no_overlap].drop(columns=['end'])

    gpu_count = (
        df_compute
        .groupby(['name', '_ki'])['gpu']
        .transform('count')
    )

    n_gpus = df['gpu'].nunique()
    return df_compute[gpu_count == n_gpus].copy()


def get_straggler_gpus(
    pytorch_traces,
    max_adj,
    invert=True,
    max_lead=0,
    use_sum=False,
    use_max=False,
    use_last=False,
):
    assert (use_max and not (use_last or use_sum)) or (use_last and not (
        use_max or use_sum)) or (use_sum and not (use_max or use_last)), "pick one to use"
    df = merge_traces(pytorch_traces)
    lv = leader_value(df, agg=True, use_last=use_last,
                      use_max=use_max, use_sum=use_sum)

    gpu_leads = lv.groupby('gpu')['lead'].sum().reset_index()
    max_lead = max(max_lead, gpu_leads['lead'].max())
    value = ((gpu_leads['lead'] - gpu_leads['lead'].min()) /
             (gpu_leads['lead'].max() - gpu_leads['lead'].min()))

    if invert:
        value = 1 - value

    lv['freq_inc'] = (
        value
        * gpu_leads['lead'].max()/max_lead
        * max_adj
    )

    return lv.set_index('gpu')['freq_inc'].to_dict(), max_lead


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from argparse import ArgumentParser

#     parser = ArgumentParser()
#     parser.add_argument(
#         "--traces",
#         "-t",
#         action="append",
#         type=str,
#         required=True,
#         nargs="+",
#     )
#     args = parser.parse_args()

#     def main(
#         traces: list[list[str]],
#     ):
#         # merge pickle
#         step_dfs = tuple(merge_traces(t) for t in zip(*traces))
#         df = pd.concat(
#             step_dfs[1:],
#             ignore_index=True
#         )

#         n_gpus = df['gpu'].nunique()
#         lv = leader_value(df, agg=False)
#         lv_sum = leader_value(df, agg=True, use_sum=True)
#         assert n_gpus == 8, f"got {
#             len(traces)} traces but expected 8. Comment this out if expected"

#         n_rows = 2
#         n_cols = 4
#         fig, axs = plt.subplots(n_rows, n_cols)
#         taxs = np.empty_like(axs, dtype=axs.dtype)
#         for row in range(n_rows):
#             for col in range(n_cols):
#                 taxs[row][col] = axs[row][col].twinx()

#         gpus = lv['gpu'].unique()
#         max_ylim0 = 0
#         min_ylim0 = 0
#         max_ylim1 = 0
#         min_ylim1 = 0
#         for gpu in gpus:
#             lv_mask = lv['gpu'] == gpu
#             lv_sum_mask = lv_sum['gpu'] == gpu
#             ax0 = axs[gpu % 2][gpu//2]
#             ax1 = taxs[gpu % 2][gpu//2]
#             ax0.scatter(
#                 lv[lv_mask].reset_index().index,
#                 lv.loc[lv_mask, 'lead'],
#                 s=.5,
#                 color='black',
#             )
#             width = lv[lv_mask].reset_index().index.values[-1]
#             ax1.bar(
#                 width/2,
#                 lv_sum.loc[lv_sum_mask, 'lead'],
#                 width=width,
#                 alpha=.5,
#                 color='purple',
#             )
#             ax1.set_zorder(0)
#             ax0.set_zorder(1)
#             ax0.patch.set_visible(False)
#             ylim0 = ax0.get_ylim()
#             ylim1 = ax1.get_ylim()
#             min_ylim0 = min(min_ylim0, ylim0[0])
#             max_ylim0 = max(max_ylim0, ylim0[1])
#             min_ylim1 = min(min_ylim1, ylim1[0])
#             max_ylim1 = max(max_ylim1, ylim1[1])

#         for gpu in gpus:
#             ax0 = axs[gpu % 2][gpu//2]
#             ax1 = taxs[gpu % 2][gpu//2]
#             if gpu % 2 == 0:
#                 ax0.set_xticklabels([])
#                 ax0.tick_params(axis='x', length=0)
#             if gpu // 2 != 0:
#                 ax0.set_yticklabels([])
#                 ax0.tick_params(axis='y', length=0)
#             else:
#                 ax0.set_ylabel('Lead Value')

#             if gpu // 2 != n_cols-1:
#                 ax1.set_yticklabels([])
#                 ax1.tick_params(axis='y', length=0)
#             else:
#                 ax1.set_ylabel('Lead Sum')
#             ax0.set_ylim((min_ylim0, max_ylim0))
#             ax1.set_ylim((min_ylim1, max_ylim1))
#         fig.tight_layout()
#         fig.savefig('straggler_detection.pdf')
#     main(args.traces)
