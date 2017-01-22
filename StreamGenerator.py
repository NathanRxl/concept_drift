class StreamGenerator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    # TODO : Use direcly StreamGenerator as an iterator instead
    def generate(self, stream_length=1e8, batch=1):
        X, y = self.data_loader.return_data()
        X_length = len(X)
        if stream_length > X_length:
            stream_length = X_length

        for i in range(0, stream_length, batch):
            yield X[i:i + batch], y[i:i + batch]
