# insider_module/xgb_model.py

import os
import joblib
import shap
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)


MODEL_PATH = "models/insider_xgb_model.pkl"


class InsiderXGBoost:

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        self.explainer = None
        self.feature_names = None

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, X, y):

        self.feature_names = list(X.columns)

        # Train/test split BEFORE SMOTEENN
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print("\nOriginal class distribution:")
        print(y_train.value_counts())

        # -----------------------------------------------------
        # SMOTEENN
        # -----------------------------------------------------

        smote_enn = SMOTEENN(
            random_state=42
        )

        X_train_balanced, y_train_balanced = smote_enn.fit_resample(
            X_train,
            y_train
        )

        print("\nAfter SMOTEENN:")
        print(pd.Series(y_train_balanced).value_counts())

        # -----------------------------------------------------
        # XGBOOST
        # -----------------------------------------------------

        self.model.fit(
            X_train_balanced,
            y_train_balanced
        )

        # -----------------------------------------------------
        # EVALUATION
        # -----------------------------------------------------

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print("\n========== XGBOOST RESULTS ==========")
        print("Accuracy:", round(accuracy, 4))
        print("F1 Score:", round(f1, 4))

        try:
            auc = roc_auc_score(
                y_test,
                probabilities
            )
            print("ROC-AUC:", round(auc, 4))
        except Exception:
            auc = None
            print("ROC-AUC: Could not calculate")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        # -----------------------------------------------------
        # SHAP
        # -----------------------------------------------------

        self.explainer = shap.TreeExplainer(
            self.model
        )

        return {
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": auc
        }

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    def predict(self, X):

        probability = self.model.predict_proba(
            X
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        return prediction, probability

    # ---------------------------------------------------------
    # SHAP EXPLANATION
    # ---------------------------------------------------------

    def explain(self, X):

        if self.explainer is None:
            self.explainer = shap.TreeExplainer(
                self.model
            )

        shap_values = self.explainer.shap_values(X)

        explanations = []

        for i in range(len(X)):

            values = shap_values[i]

            feature_values = X.iloc[i]

            explanation = {}

            for feature, value, shap_value in zip(
                self.feature_names,
                feature_values,
                values
            ):

                explanation[feature] = {
                    "value": float(value),
                    "shap_value": float(shap_value)
                }

            explanations.append(
                explanation
            )

        return explanations

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(self):

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        joblib.dump(
            self,
            MODEL_PATH
        )

        print(
            f"\nModel saved to: {MODEL_PATH}"
        )

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    @staticmethod
    def load():

        return joblib.load(
            MODEL_PATH
        )