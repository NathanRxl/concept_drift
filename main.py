from sklearn.svm import SVC
from StreamGenerator import StreamGenerator
from DataLoader import SEALoader
from AlgorithmsComparator import AlgorithmsComparator
from ensemble_methods import SEA

# generate data
loader = SEALoader('../data/sea.data')
generator = StreamGenerator(loader)

# models
SEA1 = SEA(10)
SEA2 = SEA(10, base_estimator=SVC())
algorithms = [
    ("SEA (Decision Tree)", SEA1),
    ("SEA (SVC)", SEA2)
]

# comparison of algorithms
comparator = AlgorithmsComparator(algorithms, generator)
comparator.plot_comparison(batch_size=5000)
