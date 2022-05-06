import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=1331)

# Average CV score on the training set was: -256.0447252776591
exported_pipeline = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, 
    loss="ls", max_depth=6, max_features=0.15000000000000002, min_samples_leaf=3, 
    min_samples_split=15, n_estimators=100, subsample=0.6000000000000001)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.15000000000000002, 
    min_samples_leaf=4, min_samples_split=2, n_estimators=100)),
    RandomForestRegressor(bootstrap=True, max_features=0.3, min_samples_leaf=4, m
    in_samples_split=6, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 1331)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
