import pandas as pd


class Loader:

    def __init__(self, path):
        self.path = path

    def return_data(self):
        return self.X, self.y


class SEALoader(Loader):

    def __init__(self, path):
        Loader.__init__(self, path)
        df = pd.read_csv(self.path, header=None, names=['attribute_1', 'attribute_2', 'attribute_3', 'label'])
        data = df.values
        self.X = data[:, :3]
        self.y = data[:, -1]


class Generator:

    def __init__(self, loader):
        self.loader = loader

    def generate(self, limit=1e8, batch=1):
        X, y = self.loader.return_data()
        size = len(X)
        if limit > size:
            limit = size

        for i in range(0, limit, batch):
            yield X[i:i+batch], y[i:i+batch]





