from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from AlgorithmsComparator import AlgorithmsComparator
from ensemble_methods import SEA, OnlineBagging, DDD
from drift_detection_methods.spc import DDM
from data_management.DataLoader import SEALoader, KDDCupLoader
from data_management.StreamGenerator import StreamGenerator
from training_windows_methods import AdaptiveSVC

# models
SEA_decision_trees = SEA(10, base_estimator=DecisionTreeClassifier())
SEA_SVC = SEA(10, base_estimator=SVC())
adaptive_SVC = AdaptiveSVC(C=100, memory_limit=15000)

n_classes = np.array(range(0, 2))
bagging_high_diversity = OnlineBagging(lambda_diversity=0.1, n_classes=n_classes, n_estimators=25)
bagging_low_diversity = OnlineBagging(lambda_diversity=1, n_classes=n_classes, n_estimators=25)

clf = OnlineBagging
PARAM_LOG_REG = {'solver': 'sag', 'tol': 1e-1, 'C': 1e4}
n_classes = np.array(range(0, 2))
p_clf_high = {'lambda_diversity': 0.1,
              'n_classes': n_classes,
              'n_estimators': 25,
              'base_estimator': LogisticRegression,
              'p_estimators': PARAM_LOG_REG}
p_clf_low = {'lambda_diversity': 1,
             'n_classes': n_classes,
             'n_estimators': 25,
             'base_estimator': LogisticRegression,
             'p_estimators': PARAM_LOG_REG}
ddd = DDD(ensemble_method=clf, drift_detector=DDM, pl=p_clf_low, ph=p_clf_high)

algorithms = [
    ("SEA (Decision Tree)", SEA_decision_trees),
    # ("SEA (SVC)", SEA_SVC),
    # ("Adaptive SVC", adaptive_SVC),
    ("Bagging low div (LogReg)", bagging_low_diversity),
    ("Bagging high div (LogReg)", bagging_high_diversity),
    ("DDD", ddd)
]

# generate SEA concepts data
sea_loader = SEALoader('data/sea.data', percentage_historical_data=0.2)
sea_generator = StreamGenerator(sea_loader)

# comparison of algorithms on SEA concepts
print("\nDataset: SEA concepts")
comparator = AlgorithmsComparator(algorithms, sea_generator)
comparator.plot_comparison(batch_size=3000, stream_length=480000)

# # generate KDD data
# kdd_loader = KDDCupLoader('data/kddcup.data_10_percent', percentage_historical_data=0.2)
# kdd_generator = StreamGenerator(kdd_loader)
#
# # comparison of algorithms on KDD dataset
# print("\nDataset: KDD")
# comparator = AlgorithmsComparator(algorithms, kdd_generator)
# comparator.plot_comparison(batch_size=3000)
