import unittest
from htsexperimentation.visualization.plotting_subsampled_fitting import plot_predictions_vs_original


class TestModel(unittest.TestCase):
    def setUp(self):
        self.dataset = "m5"
        self.freq = 'w'

    def test_plot_predictions(self):
        plot_predictions_vs_original('./results/', self.dataset, self.freq, sample_perc=0.75)