from torch.utils.data.sampler import Sampler
import numpy as np

class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last=False, min_size=256, max_height=480, max_width=640, size_int=8):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hmin = min_size
        self.hmax = max_height
        self.wmin = min_size
        self.wmax = max_width
        self.size_int = size_int
        self.hint = (self.hmax-self.hmin)//self.size_int+1
        self.wint = (self.wmax-self.wmin)//self.size_int+1

    def generate_height_width(self):
        hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
