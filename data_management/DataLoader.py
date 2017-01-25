import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
    def __init__(self, data_path, percentage_historical_data=0.2):
        self.data_path = data_path
        self.percentage_historical_data = percentage_historical_data
        self.X = None
        self.y = None
        self.X_historical = None
        self.y_historical = None

    def return_data(self):
        """
        The data which is used for the streaming part emulation.
        :return: Tuple X and y
        """
        return self.X, self.y

    def return_historical_data(self):
        """
        The historical used for training the model before going online.
        :return:
        """
        return self.X_historical, self.y_historical

    def split_data(self):
        """
        Split the dataset based on the percentage given in argument (percentage_historical_data)
        """
        number_histocal_data = int(self.percentage_historical_data * len(self.X))
        self.X_historical = self.X[:number_histocal_data]
        self.y_historical = self.y[:number_histocal_data]
        self.X = self.X[number_histocal_data + 1:]
        self.y = self.y[number_histocal_data + 1:]

    def normalization(self):
        """
        Normalized the data based on the historical data. Since we study concept drift we prefer to use a MinMax
        normalisation.
        """
        mms = MinMaxScaler()
        self.X_historical = mms.fit_transform(self.X_historical)
        self.X = mms.transform(self.X)

    def save_data(self, path):
        if not os.path.exists(path):
            with open(self.data_path, 'wb') as data_file:
                data = {'X': self.X, 'y': self.y, 'X_historical': self.X_historical, 'y_historical': self.y_historical}
                pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self):
        with open(self.data_path, 'rb') as data_file:
            data = pickle.load(data_file)
            self.X = data['X']
            self.y = data['y']
            self.X_historical = data['X_historical']
            self.y_historical = data['y_historical']


class SEALoader(DataLoader):
    def __init__(self, sea_data_path, use_pickle_for_loading=False, percentage_historical_data=0.2):
        DataLoader.__init__(self, sea_data_path, percentage_historical_data=percentage_historical_data)
        if use_pickle_for_loading:
            self.load_from_pickle()
        else:
            sea_df = pd.read_csv(self.data_path, header=None, names=HEADER_NAMES['SEA'])
            sea_data = sea_df.values
            self.X = sea_data[:, 1:3]
            self.y = sea_data[:, -1]
            DataLoader.split_data(self)
            DataLoader.normalization(self)


            # Normalization
            mms = MinMaxScaler()
            self.X = mms.fit_transform(self.X)




class KDDCupLoader(DataLoader):
    def __init__(self, kdd_data_path, use_pickle_for_loading=False, percentage_historical_data=0.2, dummies=True):
        '''

        :param kdd_data_path:
        :param use_pickle_for_loading: You have registered a pickle file
        :param percentage_historical_data: Percentage of data to use for the historical training.
        :param dummies: If true convert categorical variable into dummy/indicator variables (one-hot encoded).
        Use dummies equal false when your learning algorithm is DecisionTree.
        :return:
        '''
        DataLoader.__init__(self, kdd_data_path, percentage_historical_data=percentage_historical_data)
        if use_pickle_for_loading:
            self.load_from_pickle()
        else:  # TODO shorten the following lines of code
            kdd_df = pd.read_csv(
                    self.data_path,
                    index_col=False,
                    delimiter=',',
                    header=None,
                    names=HEADER_NAMES['KDD']
            )
            # TODO (minor) : Do not load these 2 columns at first
            useless_features = ["num_outbound_cmds", "is_host_login"]
            kdd_df = kdd_df.drop(useless_features, axis=1)

            # Handle symbolic data
            symbolic = [
                "protocol_type",
                "service",
                "flag",
                "label"
            ]

            self.symbolic_df = kdd_df[symbolic]
            if dummies:
                symbolic_df_without_label = self.symbolic_df[self.symbolic_df.columns.difference(['label'])]
                dummies_df = pd.get_dummies(symbolic_df_without_label)
                non_categorical = kdd_df[kdd_df.columns.difference(symbolic)].values
                # Create X
                self.X = np.concatenate((non_categorical, dummies_df.values), axis=1)
                # Create y
                label = self.symbolic_df['label'].values
                self.y = LabelEncoder().fit_transform(label)

                DataLoader.split_data(self)
                DataLoader.normalization(self)
            else:
                self.__encode_symbolic_df()
                kdd_df[symbolic] = self.symbolic_df
                self.X = kdd_df[kdd_df.columns.difference(['label'])].values
                self.y = kdd_df['label'].values
                DataLoader.split_data(self)

    def __encode_symbolic_df(self):
        self.symbolic_encoder = defaultdict(LabelEncoder)
        # Encode the symbolic variables
        self.symbolic_df = self.symbolic_df.apply(lambda x: self.symbolic_encoder[x.name].fit_transform(x))

    def inverse_encode_symbolic_df(self):
        self.symbolic_df.apply(lambda x: self.symbolic_encoder[x.name].inverse_transform(x))
