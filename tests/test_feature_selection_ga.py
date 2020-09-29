import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from feature_selection_ga import FeatureSelectionGA, FitnessFunction

MODEL = LogisticRegression()


@pytest.mark.parametrize(
    "model, features, target",
    [(MODEL, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 1, 1]))],
)
def test_feature_selection_ga(model, features, target):
    fsga = FeatureSelectionGA(model, features, target, ff_obj=FitnessFunction(2))
    result = fsga.generate(5)
    assert 5 == len(result)
