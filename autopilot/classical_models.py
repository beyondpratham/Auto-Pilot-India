import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from autopilot.config import IMAGE_SIZE, RANDOM_STATE
from autopilot.features import extract_hog_features


def build_classical_models(random_state=RANDOM_STATE):
    return {
        "hog_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True, random_state=random_state)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "gaussian_nb": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        max_features="sqrt",
                        criterion="gini",
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


class HogClassifierAdapter:
    def __init__(self, pipeline, class_names, image_size=IMAGE_SIZE):
        self.pipeline = pipeline
        self.class_names = class_names
        self.image_size = image_size

    def predict_proba_images(self, images):
        features = extract_hog_features(images, self.image_size)
        return self.pipeline.predict_proba(features)

    def save(self, path):
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "class_names": self.class_names,
                "image_size": self.image_size,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        return cls(payload["pipeline"], payload["class_names"], payload["image_size"])


def train_and_evaluate_classical_models(x_train, y_train, x_test, class_names):
    train_features = extract_hog_features(x_train)
    test_features = extract_hog_features(x_test)

    models = build_classical_models()
    trained = {}
    predictions = {}
    for name, pipeline in models.items():
        pipeline.fit(train_features, y_train)
        predictions[name] = pipeline.predict(test_features)
        trained[name] = HogClassifierAdapter(pipeline, class_names)

    return trained, predictions
