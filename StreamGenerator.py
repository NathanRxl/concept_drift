class StreamGenerator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def generate(self, stream_length=1e8, batch_size=1):
        X, y = self.data_loader.return_data()
        X_length = len(X)
        if stream_length > X_length:
            stream_length = X_length

        for i in range(0, stream_length, batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]
