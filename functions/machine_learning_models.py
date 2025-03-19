from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

ml_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "Light Gradient Boosting Machine (LGBM)": LGBMClassifier(random_state=42),
}

ml_models_parameters = {
    'Logistic Regression': {
        'model__C': [0.01, 0.25, 0.75, 1, 1.25],
        'model__penalty': ['l1', 'l2'],
        'model__max_iter': [50, 100, 150],
        'model__class_weight': ['balanced', None]
    },
    'Random Forest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__class_weight': ['balanced', None]
    },
    'Gradient Boosting': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_depth': [3, 5, 7, 10],
        'model__learning_rate': [0.1, 0.01, 0.001],
        'model__subsample': [0.5, 0.7, 1]
    },
    "K-Nearest Neighbors (KNN)": {
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance'],
        'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    "Light Gradient Boosting Machine (LGBM)": {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [-1, 10, 20],
        'model__num_leaves': [31, 50, 100],
        'model__subsample': [0.5, 0.7, 1.0]
    }
}