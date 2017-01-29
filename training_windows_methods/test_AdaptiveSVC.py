import unittest

from data_management.DataLoader import SEALoader
from data_management.StreamGenerator import StreamGenerator
from training_windows_methods.AdaptiveSVC import AdaptiveSVC


class TestAdaptiveSVC(unittest.TestCase):

    def setUp(self):
        self.sea_loader = SEALoader('data/sea.data')
        self.sea_generator = StreamGenerator(self.sea_loader)

    def test_memory_manager(self):
        # model
        clf = AdaptiveSVC(memory_limit=500, C=100)

        for i, (X, y) in enumerate(self.sea_generator.generate(batch=200)):
            if i < 5:
                clf.update(X, y)
                if i == 0:
                    self.assertEqual(clf.previous_best_window, 1)
                    self.assertEqual(clf.memory_current_size, 200)
                    self.assertEqual(len(clf.memory['y']), 1)
                if i == 1:
                    self.assertEqual(clf.previous_best_window, 2)
                    self.assertEqual(clf.memory_current_size, 400)
                    self.assertEqual(len(clf.memory['y']), 2)
                if i == 2:
                    self.assertEqual(clf.previous_best_window, 3)
                    self.assertEqual(clf.memory_current_size, 500)
                    self.assertEqual(len(clf.memory['y']), 3)
                if i == 3:
                    self.assertEqual(clf.previous_best_window, 3)
                    self.assertEqual(clf.memory_current_size, 500)
                    self.assertEqual(len(clf.memory['y']), 3)
                if i == 4:
                    self.assertEqual(clf.previous_best_window, 3)
                    self.assertEqual(clf.memory_current_size, 500)
                    self.assertEqual(len(clf.memory['y']), 3)
            else:
                break

if __name__ == '__main__':
    unittest.main()
