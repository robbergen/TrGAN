import chainer
import numpy as np


class HecktorDataset(chainer.dataset.DatasetMixin):

    def __init__(self, n_frames, dataset_path):
        self.dset = np.load(dataset_path)
        if np.ndim(self.dset)<5: #If there is no conditional mask
            self.dset = self.dset.transpose([1, 0, 2, 3])
            self.dset = self.dset[:,:,np.newaxis,:,:]
        else:
            self.dset = self.dset.transpose([1, 0, 4, 2, 3]) #If there is a conditional mask there is no need for a new axis
        self.n_frames = n_frames

    def __len__(self):
        return self.dset.shape[0]

    def get_example(self, i):
        T = self.dset.shape[1]
        ot = np.random.randint(T - self.n_frames) if T > self.n_frames else 0
        x = self.dset[i, ot:(ot + self.n_frames)]
        return np.asarray((x - 128.0) / 128.0, dtype=np.float32)
