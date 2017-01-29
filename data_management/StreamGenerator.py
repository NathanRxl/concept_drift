class StreamGenerator:
    """
    Emulate a stream of data for online learning algorithm
    """
    def __init__(self, data_loader):
        """
        Constructor of the StreamGenerator
        :param data_loader: Loader which inherits DataLoader
        """
        self.data_loader = data_loader

    def get_historical_data(self):
        """
        :return: A tuple X_historical_data and y_historical_data
        """
        return self.data_loader.return_historical_data()

    def generate(self, stream_length=1e8, batch_size=1):
        """
        Generator of streaming data
        :param stream_length: How many example do you want to see
        :param batch_size: batch size is one by default you can fixed it or randomized it.
        :return: A tuple X,y
        """
        X, y = self.data_loader.return_data()
        X_length = len(X)
        if stream_length > X_length:
            stream_length = X_length

        for i in range(0, stream_length, batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]
