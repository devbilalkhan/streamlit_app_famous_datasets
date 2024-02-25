from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
import random


models_dict = {
    'Classification': {
        'Logistic Regression': LogisticRegression(C=random.uniform(0.01, 1.0), max_iter=100, solver='lbfgs'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
        'Support Vector Machine': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True),
        'DecisionTree': DecisionTreeClassifier(max_depth=None, criterion='gini'),
        'RandomForest': RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=random.uniform(0.01, 2.0)),
        'CatBoost': CatBoostClassifier(iterations=1000, learning_rate=random.uniform(0.01, 0.3), depth=6, verbose=False),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=random.uniform(0.01, 0.3), use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(num_leaves=31, learning_rate=random.uniform(0.01, 0.3), n_estimators=100),
    },
    'Regression': {
        'Linear Regression': LinearRegression(fit_intercept=True),
        'Support Vector Machine': SVR(C=1.0, epsilon=0.1, kernel='rbf'),
        'DecisionTree': DecisionTreeRegressor(max_depth=None),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=None),
        'AdaBoost': AdaBoostRegressor(n_estimators=50, learning_rate=random.uniform(0.01, 2.0), loss='linear'),
        'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=random.uniform(0.01, 0.3), depth=6, verbose=False),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=random.uniform(0.01, 0.3)),
        'LightGBM': LGBMRegressor(num_leaves=31, learning_rate=random.uniform(0.01, 0.3), n_estimators=100),
    }
}