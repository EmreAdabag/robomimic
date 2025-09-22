import argparse
import os
from typing import List

import h5py
import numpy as np


def compute_sparse_indices(num_samples: int, step: int) -> List[int]:
    assert step >= 1
    idx = list(range(0, num_samples, step))
    if len(idx) == 0 or idx[-1] != num_samples - 1:
        idx.append(num_samples - 1)
    return idx


def sparse_sample_dataset(in_path: str, out_path: str, step: int) -> None:
    """
    Create a sparsely-subsampled copy of a robomimic hdf5 dataset.

    - Keep every `step` frame per trajectory, always including the final frame
    - Remap next_obs so that next_obs[i] == obs[i+1] in the new dataset (for i < N-1)
      and next_obs[N-1] is copied from the original next_obs at the final kept index

    Keys handled per demo:
    - obs/* (all subkeys)
    - next_obs/* (rebuilt as described)
    - actions (if present)
    - rewards (if present)
    - dones (if present)
    - states (if present)
    Other keys are ignored.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        assert 'data' in fin, "Input hdf5 missing 'data' group"
        grp_in_data = fin['data']
        grp_out_data = fout.create_group('data')

        # copy dataset-level attrs (e.g., env_args) to output
        for attr_key in grp_in_data.attrs.keys():
            grp_out_data.attrs[attr_key] = grp_in_data.attrs[attr_key]

       
        for demo_key in grp_in_data.keys():
            g_in = grp_in_data[demo_key]
            g_out = grp_out_data.create_group(demo_key)

            num_samples = int(g_in.attrs['num_samples'])
            idx = compute_sparse_indices(num_samples, step)
            N = len(idx)

            # copy attrs
            # copy all attributes to g_out
            for attr_key in g_in.attrs.keys():
                g_out.attrs[attr_key] = g_in.attrs[attr_key]
                # print(f"Copied attr: {attr_key}")
            # update num_samples to reflect the sparse-sampled count
            g_out.attrs['num_samples'] = N

            # obs subkeys
            if 'obs' in g_in:
                obs_in = g_in['obs']
                obs_out = g_out.create_group('obs')
                obs_keys = list(obs_in.keys())
                # materialize obs first for next_obs rebuild
                obs_sparse = {}
                for k in obs_keys:
                    arr = obs_in[k][()]
                    obs_sparse[k] = arr[idx]
                    obs_out.create_dataset(k, data=obs_sparse[k])
            else:
                obs_sparse = {}

            # actions / rewards / dones / states
            for ds_name in ['actions', 'rewards', 'dones', 'states']:
                if ds_name in g_in:
                    arr = g_in[ds_name][()]
                    g_out.create_dataset(ds_name, data=arr[idx])

            # next_obs: rebuild to match next sampled step
            # next_obs[i] = obs_sparse[i+1] for i < N-1; last uses original next_obs at last kept index
            if len(obs_sparse) > 0:
                next_obs_out = g_out.create_group('next_obs')
                last_src_idx = idx[-1]
                if 'next_obs' in g_in:
                    next_obs_in = g_in['next_obs']
                else:
                    next_obs_in = None

                for k in obs_sparse.keys():
                    # shape [N, ...]
                    tgt = np.empty_like(obs_sparse[k])
                    if N > 1:
                        tgt[:-1] = obs_sparse[k][1:]
                    # last element
                    if next_obs_in is not None and k in next_obs_in:
                        last_next = next_obs_in[k][()][last_src_idx]
                    else:
                        # fallback: duplicate last obs
                        last_next = obs_sparse[k][-1]
                    tgt[-1] = last_next
                    next_obs_out.create_dataset(k, data=tgt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_path', type=str, required=True, help='Input hdf5 path')
    parser.add_argument('--out', dest='out_path', type=str, required=True, help='Output hdf5 path')
    parser.add_argument('--step', dest='step', type=int, required=True, help='Keep every n-th frame; always include last')
    args = parser.parse_args()

    sparse_sample_dataset(args.in_path, args.out_path, args.step)


if __name__ == '__main__':
    main()


