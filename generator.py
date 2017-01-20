import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


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


class KDDCupLoader(Loader):
    def __init__(self, path):
        Loader.__init__(self, path)
        headers = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
                   "urgent",
                   "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                   "is_guest_login", "count",
                   "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                   "diff_srv_rate",
                   "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                   "dst_host_diff_srv_rate",
                   "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                   "dst_host_srv_serror_rate",
                   "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

        df = pd.read_csv(self.path, index_col=False, delimiter=',', header=None, names=headers)

        # Symbolic data
        symbolic = ["protocol_type", "service", "flag", "land", "logged_in", "is_host_login", "is_guest_login", "label"]
        useless_features = ["num_outbound_cmds", "is_host_login"]
        df.drop(useless_features, axis=1)
        self.symbolic_df = df[symbolic]
        self.__encoding_symbolic_df()
        df[symbolic] = self.symbolic_df
        self.X = df[df.columns.difference(['label'])].values
        self.y = df['label'].values

    # TODO handles more properly fit
    def __encoding_symbolic_df(self):
        '''
        We encode on all the data beacause we suppose that we have already checked all the possible
        that t
        :param df: Dataframe which for each column has symbolic value (strings...)
        '''
        self.dico = defaultdict(LabelEncoder)
        # Encoding the variable
        self.symbolic_df = self.symbolic_df.apply(lambda x: self.dico[x.name].fit_transform(x))

    def __inverse_encoding_df(self):
        self.symbolic_df.apply(lambda x: self.dico[x.name].inverse_transform(x))


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





