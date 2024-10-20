# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from torchvision import transforms

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        images = None
        if len(self._image_fnames) == 0:
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in ".npz")
            if len(self._image_fnames) == 0:
                raise IOError('No image files found in the specified path')
            else:
                images = np.concatenate([np.load(os.path.join(self._path, fname))["samples"].transpose(0, 3, 1, 2) for fname in self._image_fnames], axis = 0)
                raw_shape = images.shape
                super_kwargs['cache'] = True
        else:
            raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        name = os.path.splitext(os.path.basename(self._path))[0]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

        if images is not None:
            self._cached_images = {i: images[i] for i in range(images.shape[0])}

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        # print(fname)
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

class optimal_denoiser_dataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        transforms = transforms.Compose([]),
    ):
        self._path = path
        self._file_list = os.listdir(self._path)
        self._file_list.sort()
        self.images = [torch.load(os.path.join(self._path, pth)) for pth in self._file_list]
        self.images = torch.cat(self.images).permute((0, 3, 1, 2))
        
        # if select_image_num is not None and select_image_num > 0:
        #     self.images = self.images[:select_image_num]
        self.transforms = transforms
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        return image

class MoLRG(torch.utils.data.Dataset):
    def __init__(self,
        resolution = 2,
        class_num = 2,
        per_class_dim = 2,
        sample_per_class = 500,
        path = "./datasets",
        use_labels = False,
        save_dataset = True,
        loading_dataset = True,
    ):  
        img_resolution = torch.tensor([resolution, resolution, 3])
        dataset_path = os.path.join(path, f"MoLRG_dataset_resolution{resolution}_classnum{class_num}_perclassdim{per_class_dim}_sample{sample_per_class}.pt")
        if (not os.path.exists(dataset_path)) or (not loading_dataset):
            print("Create new dataset......")
            dim = img_resolution.prod()
            rand = torch.randn(dim, dim)
            U, _, _ = torch.linalg.svd(rand)
            classbasis = []
            ## generate basis
            for i in range(class_num):
                classbasis.append(U[:, per_class_dim * i:per_class_dim * (i+1)][None, :])
            classbasis = torch.cat(classbasis)

            ## generate sample
            data = []
            conds = []
            for cond in range(class_num):
                for idx in range(sample_per_class):
                    data.append((classbasis[cond] @ torch.randn (per_class_dim, 1)).reshape((1, resolution, resolution, 3)))
                    conds.append(cond)
            data = torch.cat(data)
            conds = torch.tensor(conds)
            if save_dataset:
                torch.save({
                    "basis": classbasis,
                    "space_basis": U,
                    "data":data,
                    "class_num": class_num,
                    "sample_per_class": sample_per_class,
                    "per_class_dim": per_class_dim,
                    "resolution": resolution,
                    "conds": conds,
                }, dataset_path)
        else:
            dataset = torch.load(dataset_path)
            resolution = dataset["resolution"]
            class_num = dataset["class_num"]
            per_class_dim = dataset["per_class_dim"]
            sample_per_class = dataset["sample_per_class"]  
            data = dataset["data"]    
            classbasis = dataset["basis"]      
            conds = dataset["conds"]
        self.data = data
        self.conds = conds
        self.name = "MoLRG"
        self.resolution = resolution
        self.num_channels= 3
        self.label_dim= 0
        self.basis = classbasis

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return (self.data[idx].permute((2, 0, 1)) + 1) * 127.5, self.conds[idx]
#----------------------------------------------------------------------------
