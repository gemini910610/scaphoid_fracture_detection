import math

from random import Random

class DataSplitter:
    def __init__(self, train_size=0.8, seed=42):
        self.train_size = train_size
        self.random = Random(seed)
    def shuffle(self, data):
        data = self.random.sample(data, len(data))
        return data
    def train_val_split(self, data):
        data = self.shuffle(data)

        train_count = math.ceil(len(data) * self.train_size)
        train_data = data[:train_count]
        val_data = data[train_count:]
        return train_data, val_data
