from sklearn.datasets import make_classification
from sklearn import linear_model
from feature_selection_ga import FeatureSelectionGA
import fitness_function as ff

X, y = make_classification(
    n_samples=100,
    n_features=15,
    n_classes=3,
    n_informative=4,
    n_redundant=1,
    n_repeated=2,
    random_state=1,
)

model = linear_model.LogisticRegression(solver="lbfgs", multi_class="auto")
fsga = FeatureSelectionGA(model, X, y, verbose=1, ff_obj=ff.FitnessFunction())
pop = fsga.generate(100)