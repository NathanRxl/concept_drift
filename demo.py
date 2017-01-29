from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from AlgorithmsComparator import AlgorithmsComparator
from ensemble_methods import SEA
from ensemble_methods import DWM
from data_management.DataLoader import SEALoader
from data_management.StreamGenerator import StreamGenerator
from training_windows_methods import AdaptiveSVC

# models

# SEA
SEA_decision_trees = SEA(10, base_estimator=DecisionTreeClassifier())
SEA_SVC = SEA(10, base_estimator=SVC())

# DWM
DWM_decision_trees = DWM(base_estimator=DecisionTreeClassifier(), beta = 0.5, theta = 0.01, period = 3)
DWM_SVC = DWM(base_estimator=SVC(probability = True), beta = 0.5, theta = 0.01, period = 3)

# Adaptive SVC
adaptive_SVC = AdaptiveSVC(C=100, memory_limit=15000)

algorithms = [
    ("SEA (Decision Tree)", SEA_decision_trees),
    ("SEA (SVC)", SEA_SVC),
    ("Adaptive SVC", adaptive_SVC),
]

# generate SEA concepts data
sea_loader = SEALoader('data/sea.data', percentage_historical_data=0.2)
sea_generator = StreamGenerator(sea_loader)

# comparison of algorithms on SEA concepts
print("\nDataset: SEA concepts")
comparator = AlgorithmsComparator(algorithms, sea_generator)
comparator.plot_comparison(batch_size=3000, stream_length=48000)

# # generate KDD data
# kdd_loader = KDDCupLoader('data/kddcup.data_10_percent', percentage_historical_data=0.2)
# kdd_generator = StreamGenerator(kdd_loader)
#
# # comparison of algorithms on KDD dataset
# print("\nDataset: KDD")
# comparator = AlgorithmsComparator(algorithms, kdd_generator)
# comparator.plot_comparison(batch_size=3000)
