import torch
import numpy as np
import h5py
import random
from pathlib import Path
from copy import deepcopy
from utils import compute_F, qvec2rotmat, randquat
from collections.abc import Iterable
import contextlib


class CompoundDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets, name=None):
        super().__init__()
        self.datasets = datasets
        self.name = name
        self.iters_done = 0

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __iter__(self):
        self.iters_done += 1
        iternum = self.iters_done
        iters = [iter(d) for d in self.datasets]
        nm = 0
        i = 0
        while len(iters) > 0:
            i = (i + 1) % len(iters)
            it = iters[i]
            try:
                yield next(it)
                nm += 1
            except StopIteration:
                iters.remove(it)

class PoseEstimationDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_source, maxlen=None, deterministic=False):
        self.src = data_source
        self.ll = min(maxlen, len(data_source)) if maxlen is not None else len(data_source)
        self.itrno = 0
        self.deterministic = deterministic

    def __new__(cls, data_source, *args, **kwargs):
        if isinstance(data_source, Iterable):
            return CompoundDataset([PoseEstimationDataset(ds, *args, **kwargs) for ds in data_source])
        return super(PoseEstimationDataset, cls).__new__(cls)

    def __len__(self):
        return self.ll

    def __iter__(self):
        ii = deepcopy(self.src.pairs_index)
        rng = random.Random(self.itrno + 1)
        if not self.deterministic:
            self.itrno += 1
        rng.shuffle(ii)
        success = 0
        # ii = np.random.choice(self.image_indices, self.ll, replace=False)
        for i1, i2 in ii:
            if success >= self.ll:
                break
            # T_i2_i1 = T_i2_w @ T_w_i1
            RT_gt = self.src.rt(i2) @ np.linalg.inv(self.src.rt(i1))
            RT_gt = RT_gt[:3]
            R_gt = RT_gt[:3, :3]
            T_gt = RT_gt[:3, -1]
            K1, K2 = self.src.K(i1), self.src.K(i2)
            K1inv, K2inv = map(np.linalg.inv, (K1, K2))
            kp1, kp2, sides = self.src.match(i1, i2)
            MAX_KP = 20000
            MIN_KP = 8
            if len(kp1) > MAX_KP:
                keep_idx = np.random.choice(len(kp1), MAX_KP, replace=False)
                kp1, kp2 = map(lambda t: t[keep_idx], (kp1, kp2))
                if sides is not None:
                    sides = sides[keep_idx]
            elif len(kp1) < MIN_KP:
                continue

            t = lambda x: torch.tensor(x, dtype=torch.float32, device='cpu')
            data_list = [t(self.unproj(kp1, K1inv)), t(self.unproj(kp2, K2inv))]
            success += 1
            K1K2 = (K1.astype(np.float32), K2.astype(np.float32))
            out = {'points': torch.cat(data_list, dim=-1),
                   'sides': t(sides),
                   'RT_gt': t(RT_gt),
                   'K1K2': K1K2}
            yield out

    @staticmethod
    def unproj(k, Kinv):
        return k @ Kinv[:2, :2] + Kinv[:2, 2]

@contextlib.contextmanager
def np_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class SyntheticDataIndex:
    def __init__(self, ll=10000):
        self.pairs_index = list(zip(np.random.randint(0, 100000, ll), np.random.randint(0, 10000, ll)))
        self.ll = len(self.pairs_index)
        self.inlier_rate_min = 0.05
        self.inlier_rate_max = 0.4
        self.num_points = 2000
        self.pix_noise_magnitude = 0.5
        self.rt_cache = {}

    def __len__(self):
        return self.ll

    def rt(self, i):
        if i in self.rt_cache:
            return self.rt_cache[i]
        with np_temp_seed(i):
            R = qvec2rotmat(randquat())
            T = np.array([0., 0., 2.5])
            RT = np.eye(4)
            RT[:3, :3] = R
            RT[:3, -1] = T
            self.rt_cache[i] = RT
            return RT

    def impath(self, i):
        return None

    def K(self, i):
        return np.array([[1500., 0., 750.],
                         [0., 1500., 750.],
                         [0., 0., 1.]])

    def produce_cloud(self, n):
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.random.uniform(0., 1., size=(n, 1)) ** (1/3)
        pts = pts * norms
        return pts

    def reproject(self, cloud, RT, K):
        R, T = RT[:3, :3], RT[:3, -1]
        proj = (cloud @ R.T + T) @ K.T
        assert np.all(proj > 0)
        proj_xy = proj[:, :2] / proj[:, 2:]
        return proj_xy

    def match(self, i1, i2):
        RT1 = self.rt(i1)
        K1 = self.K(i1)
        RT2 = self.rt(i2)
        K2 = self.K(i2)
        with np_temp_seed(i1 * i2):
            inlier_rate = np.random.uniform() * (self.inlier_rate_max - self.inlier_rate_min) + self.inlier_rate_min
            num_inliers = int(inlier_rate * self.num_points)
            num_outliers = self.num_points - num_inliers
            inliers = self.produce_cloud(num_inliers)
            outliers1 = self.produce_cloud(num_outliers)
            outliers2 = self.produce_cloud(num_outliers)

            inlier_k1 = self.reproject(inliers, RT1, K1)
            inlier_k2 = self.reproject(inliers, RT2, K2)
            outlier_k1 = self.reproject(outliers1, RT1, K1)
            outlier_k2 = self.reproject(outliers2, RT2, K2)
            k1 = np.concatenate([inlier_k1, outlier_k1], axis=0) + np.random.randn(self.num_points, 2) * self.pix_noise_magnitude
            k2 = np.concatenate([inlier_k2, outlier_k2], axis=0) + np.random.randn(self.num_points, 2) * self.pix_noise_magnitude
            snn_inliers = np.random.beta(2, 1, num_inliers)
            snn_outliers = np.random.beta(5, 1, num_outliers)
            snn = np.concatenate([snn_inliers, snn_outliers], axis=0)
            mscores = np.random.uniform(0., 1., self.num_points)
            sides = []
            sides.append(snn[:, None])
            sides = np.concatenate(sides, axis=1)
        return k1.astype(np.float32), k2.astype(np.float32), sides.astype(np.float32)


class PhototourismDataIndex:
    TRAIN_LIST = [
        "brandenburg_gate",
        "colosseum_exterior",
        "notre_dame_front_facade",
        "pantheon_exterior",
        "taj_mahal",
        "trevi_fountain",
        "buckingham_palace",
        "grand_place_brussels",
        "palace_of_westminster",
        "prague_old_town_square",
        "temple_nara_japan",
        "westminster_abbey"
    ]
    VAL_LIST = [
        "sacre_coeur",
        "st_peters_square"
    ]
    def __init__(self, landmark_name, dataset_dir):
        if landmark_name in PhototourismDataIndex.VAL_LIST:
            mode = "val"
        elif landmark_name in PhototourismDataIndex.TRAIN_LIST:
            mode = "train"
        else:
            raise ValueError(f"Unrecognized landmark name {landmark_name}.")
        base_dir = Path(dataset_dir) / mode / landmark_name
        self.base_dir = base_dir
        # Loading the ground truth rotation matrices
        self.R_all = h5py.File(base_dir / 'R.h5', 'r')
        # Loading the ground truth translation vectors
        self.t_all = h5py.File(base_dir / 'T.h5', 'r')
        # Loading the intrinsic camera parameters
        if not (base_dir / "K.h5").exists():
            self.make_k()
        self.K_all = h5py.File(base_dir / 'K.h5', 'r')

        self.matches = h5py.File(base_dir / 'matches.h5', 'r')
        # Loading the match confidences. This basically is the 2nd nearest neighbor ratio here.
        self.match_conf = h5py.File(base_dir / 'match_conf.h5', 'r')

        # Image pairs in the data
        image_pairs_cache = base_dir / 'image_pairs.txt'
        if image_pairs_cache.exists():
            with open(image_pairs_cache, 'r') as impc:
                self.all_pairs = impc.read().split("\n")
        else:
            self.all_pairs = tuple(self.matches.keys())
            with open(image_pairs_cache, "w") as impc:
                impc.write("\n".join(self.all_pairs))
        self.ll = len(self.all_pairs)
        self.pairs_index = [pair.split('-') for pair in self.all_pairs]


    def __new__(cls, landmark_name, *args, **kwargs):
        if isinstance(landmark_name, Iterable) and not isinstance(landmark_name, str):
            return [PhototourismDataIndex(l, *args, **kwargs) for l in landmark_name]
        return super(PhototourismDataIndex, cls).__new__(cls)

    def __len__(self):
        return self.ll

    def make_k(self):
        target = h5py.File(self.base_dir / "K.h5", "w")
        source = h5py.File(self.base_dir / "K1_K2.h5", "r")
        keys_found = {}
        for k in source.keys():
            k1, k2 = k.split("-")
            if k1 not in keys_found.keys():
                keys_found[k1] = (k, 0)
            if k2 not in keys_found.keys():
                keys_found[k2] = (k, 1)
        for k, (k1k2, i) in keys_found.items():
            target[k] = np.array(source[k1k2])[0, i]
        source.close()
        target.close()

    def rt(self, i):
        R, t = np.array(self.R_all[i]), np.array(self.t_all[i])
        assert t.shape == (3, 1)
        assert R.shape == (3, 3)
        rt = np.eye(4)
        rt[:3, :3] = R
        rt[:3, -1:] = t
        return rt

    def impath(self, i):
        return self.base_dir / 'images' / f'{i}.jpg'

    def K(self, i):
        return np.array(self.K_all[i])

    def match(self, i1, i2):
        pair = f"{i1}-{i2}"
        x1x2 = self.matches[pair]
        x1 = x1x2[:, 0:2]
        x2 = x1x2[:, 2:4]
        match_scores = np.array(self.match_conf[pair])
        return x1, x2, match_scores[..., None]

    def __del__(self):
        self.matches.close()
        self.match_conf.close()
        self.R_all.close()
        self.t_all.close()
        self.K_all.close()

