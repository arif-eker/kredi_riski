from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

rf_params = {"max_depth": [3, 5, 8],
             "max_features": [8, 15, 25],
             "n_estimators": [200, 500, 1000],
             "min_samples_split": [2, 5, 10]}

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [200, 500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

xgb_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [3, 5, 8],
              "n_estimators": [200, 500, 1000],
              "colsample_bytree": [0.7, 1]}


def get_tuned_models(x_train, y_train,rnd_state):

    rf = RandomForestClassifier(random_state=rnd_state)
    lgbm = LGBMClassifier(random_state=rnd_state)
    xgb = XGBClassifier(random_state=rnd_state)

    gs_cv_rf = GridSearchCV(rf,
                            rf_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=2).fit(x_train, y_train)

    gs_cv_lgbm = GridSearchCV(lgbm,
                              lgbm_params,
                              cv=10,
                              n_jobs=-1,
                              verbose=2).fit(x_train, y_train)

    gs_cv_xgb = GridSearchCV(xgb,
                             xgb_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(x_train, y_train)

    rf_tuned = RandomForestClassifier(**gs_cv_rf.best_params_, random_state=rnd_state).fit(x_train, y_train)

    lgbm_tuned = LGBMClassifier(**gs_cv_lgbm.best_params_, random_state=rnd_state).fit(x_train, y_train)

    xgb_tuned = XGBClassifier(**gs_cv_xgb.best_params_, random_state=rnd_state).fit(x_train, y_train)

    return rf_tuned, lgbm_tuned, xgb_tuned
