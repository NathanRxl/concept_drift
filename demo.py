from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from AlgorithmsComparator import AlgorithmsComparator
from data_management.DataLoader import SEALoader, KDDCupLoader, UsenetLoader
from data_management.StreamGenerator import StreamGenerator
from drift_detection_methods.spc import DDM
from ensemble_methods import SEA, OnlineBagging, DDD, DiversityWrapper
from offline_methods import OfflineAlgorithmsWrapper
from training_windows_methods import AdaptiveSVC

# generate SEA concepts data
# sea_loader = SEALoader('data/sea.data', percentage_historical_data=0.2)
# list_classes = sea_loader.get_classes()
# sea_generator = StreamGenerator(sea_loader)
#
# # generate KDD data
# kdd_loader = KDDCupLoader('data/kddcup.data_10_percent', percentage_historical_data=0.2)
# list_classes = kdd_loader.get_classes()
# kdd_generator = StreamGenerator(kdd_loader)

usenet_loader = UsenetLoader('data/usenet_recurrent3.3.data', percentage_historical_data=0.1)
list_classes = usenet_loader.get_classes()
usenet_generator = StreamGenerator(usenet_loader)

# models
SEA_decision_trees = SEA(10, list_classes=list_classes,
                         base_estimator=OfflineAlgorithmsWrapper(DecisionTreeClassifier()))
SEA_SVC = SEA(10, base_estimator=OfflineAlgorithmsWrapper(SVC()))
adaptive_SVC = AdaptiveSVC(C=100, memory_limit=465)
decision_tree = OfflineAlgorithmsWrapper(base_estimator=DecisionTreeClassifier())

# Online Bagging
bagging_high_diversity = OnlineBagging(lambda_diversity=0.1, n_classes=list_classes, n_estimators=25)
bagging_low_diversity = OnlineBagging(lambda_diversity=1, n_classes=list_classes, n_estimators=25)

# DDD with Sea
PARAM_LOG_REG = {'solver': 'sag', 'tol': 1e-1, 'C': 1e4}
log_high_diversity = DiversityWrapper(lambda_diversity=0.1,
                                      list_classes=list_classes,
                                      base_estimator=LogisticRegression(**PARAM_LOG_REG))
log_low_diversity = DiversityWrapper(lambda_diversity=1,
                                     list_classes=list_classes,
                                     base_estimator=LogisticRegression(**PARAM_LOG_REG))
ddd_sea_log_reg = SEA
p_sea_high_div = {
    'base_estimator': log_high_diversity,
    'n_estimators': 25,
    'list_classes': list_classes
}
p_sea_low_div = {
    'base_estimator': log_low_diversity,
    'n_estimators': 25,
    'list_classes': list_classes
}
ddd = DDD(ensemble_method=ddd_sea_log_reg, drift_detector=DDM, pl=p_sea_high_div, ph=p_sea_low_div)

# DDD with online Bagging
clf = OnlineBagging
p_clf_high = {'lambda_diversity': 0.1,
              'n_classes': list_classes,
              'n_estimators': 25,
              'base_estimator': SGDClassifier,
              'p_estimators': {'loss': 'log'}  # We cannot predict_proba with the hinge loss
              }
p_clf_low = {'lambda_diversity': 1,
             'n_classes': list_classes,
             'n_estimators': 25,
             'base_estimator': SGDClassifier,
             'p_estimators': {'loss': 'log'}  # We cannot predict_proba with the hinge loss
             }
ddd_online_bagging = DDD(ensemble_method=clf, drift_detector=DDM, pl=p_clf_low, ph=p_clf_high)

algorithms = [
    # ("SEA (Decision Tree)", SEA_decision_trees),
    ("Offline decision tree", decision_tree),
    #("SEA (SVC)", SEA_SVC),
    #("Adaptive SVC", adaptive_SVC),
    ("Bagging low div (LogReg)", bagging_low_diversity),
    #("Bagging high div (LogReg)", bagging_high_diversity),
    ("DDD (Online bagging)", ddd_online_bagging),
    #("DDD (SEA LogReg)", ddd)
]



# comparison of algorithms on SEA concepts
# print("\nDataset: SEA concepts")
# comparator = AlgorithmsComparator(algorithms, sea_generator)
# comparator.plot_comparison(batch_size=3000, stream_length=48000)

# comparison of algorithms on KDD dataset
# print("\nDataset: KDD")
# comparator = AlgorithmsComparator(algorithms, kdd_generator)
# comparator.plot_comparison(batch_size=3000, stream_length=480000)

print("\n Dataset: Usenet")
comparator = AlgorithmsComparator(algorithms, usenet_generator)
comparator.plot_comparison(batch_size=50, stream_length=6000)
