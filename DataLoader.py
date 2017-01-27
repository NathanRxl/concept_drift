import os
import pickle
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

HEADER_NAMES = {
    'SEA': [
        'attribute_1',
        'attribute_2',
        'attribute_3',
        'label'
    ],
    'KDD': [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'label'
    ],
}


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X = None
        self.y = None

    def return_data(self):
        return self.X, self.y

    def save_data(self, path):
        if not os.path.exists(path):
            with open(self.data_path, 'wb') as data_file:
                data = {'X': self.X, 'y': self.y}
                pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self):
        with open(self.data_path, 'rb') as data_file:
            data = pickle.load(data_file)
            self.X = data['X']
            self.y = data['y']


class SEALoader(DataLoader):
    def __init__(self, sea_data_path, use_pickle_for_loading=False):
        DataLoader.__init__(self, sea_data_path)
        if use_pickle_for_loading:
            self.load_from_pickle()
        else:
            sea_df = pd.read_csv(self.data_path, header=None, names=HEADER_NAMES['SEA'])
            sea_data = sea_df.values
            self.X = sea_data[:, 1:3]
            self.y = sea_data[:, -1]


class KDDCupLoader(DataLoader):
    def __init__(self, kdd_data_path, use_pickle_for_loading=False):
        DataLoader.__init__(self, kdd_data_path)
        if use_pickle_for_loading:
            self.load_from_pickle()
        else:
            kdd_df = pd.read_csv(
                self.data_path,
                index_col=False,
                delimiter=',',
                header=None,
                names=HEADER_NAMES['KDD']
            )
            # TODO (minor) : Do not load these 2 columns at first
            useless_features = ["num_outbound_cmds", "is_host_login"]
            kdd_df.drop(useless_features, axis=1)

            # Handle symbolic data
            symbolic = [
                "protocol_type",
                "service",
                "flag",
                "land",
                "logged_in",
                "is_host_login",
                "is_guest_login",
                "label"
            ]
            self.symbolic_df = kdd_df[symbolic]
            self.__encode_symbolic_df()
            kdd_df[symbolic] = self.symbolic_df

            self.X = kdd_df[kdd_df.columns.difference(['label'])].values
            self.y = kdd_df['label'].values

    def __encode_symbolic_df(self):
        self.symbolic_encoder = defaultdict(LabelEncoder)
        # Encode the symbolic variables
        self.symbolic_df = self.symbolic_df.apply(lambda x: self.symbolic_encoder[x.name].fit_transform(x))

    def inverse_encode_symbolic_df(self):
        self.symbolic_df.apply(lambda x: self.symbolic_encoder[x.name].inverse_transform(x))
