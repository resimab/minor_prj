# ============================================================
# prediction_service.py
# ============================================================

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


BASE_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIRECTORY
    / "models"
    / "restaurant_feasibility_pipeline.joblib"
)


MODEL_FEATURES = [
    # Grouped/composite features used by the trained model
    "cultural_activity_score",
    "accessibility_score",
    "competition_pressure_score",
    "education_score",
    "market_gap_score",
    "health_activity_score",
    "attraction_score",
    "commercial_score",
    # Required contextual/auxiliary features
    "search_area",
    "avg_restaurant_rating_500m",
    "avg_review_ratings_500m",
    "nearest_restaurant_m",
]


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found: {MODEL_PATH}"
    )


model_package = joblib.load(
    MODEL_PATH
)


if isinstance(model_package, dict):
    model_pipeline = model_package.get(
        "pipeline"
    )

    model_metadata = model_package.get(
        "metadata",
        {}
    )
else:
    model_pipeline = model_package
    model_metadata = {}


if model_pipeline is None:
    raise ValueError(
        "The saved model package does not contain a pipeline."
    )


# Compatibility shim: ensure LogisticRegression estimators have a
# `multi_class` attribute when missing (can occur across sklearn versions).
def _ensure_logistic_multi_class_attribute(pipeline_obj):
    try:
        if hasattr(pipeline_obj, "named_steps"):
            for step in pipeline_obj.named_steps.values():
                if step.__class__.__name__ == "LogisticRegression":
                    if not hasattr(step, "multi_class"):
                        if hasattr(step, "classes_"):
                            classes = getattr(step, "classes_")
                            step.multi_class = (
                                "multinomial" if len(classes) > 2 else "ovr"
                            )
                        else:
                            step.multi_class = "ovr"
        elif pipeline_obj.__class__.__name__ == "LogisticRegression":
            if not hasattr(pipeline_obj, "multi_class"):
                if hasattr(pipeline_obj, "classes_"):
                    classes = getattr(pipeline_obj, "classes_")
                    pipeline_obj.multi_class = (
                        "multinomial" if len(classes) > 2 else "ovr"
                    )
                else:
                    pipeline_obj.multi_class = "ovr"
    except Exception:
        pass


# Apply shim right after loading the pipeline
_ensure_logistic_multi_class_attribute(model_pipeline)


def convert_numpy_value(
    value: Any
) -> Any:
    """
    Convert NumPy values into normal Python values.
    """

    if isinstance(value, np.generic):
        return value.item()

    return value


def class_to_label(
    class_value: Any
) -> str:
    """
    Convert the model class into its readable label.
    """

    class_value = convert_numpy_value(
        class_value
    )

    metadata_mapping = model_metadata.get(
        "label_mapping"
    )

    if isinstance(metadata_mapping, dict):

        if class_value in metadata_mapping:
            return str(
                metadata_mapping[class_value]
            )

        if str(class_value) in metadata_mapping:
            return str(
                metadata_mapping[str(class_value)]
            )

    # The model may already predict textual labels.
    if str(class_value) in {
        "Low",
        "Moderate",
        "High"
    }:
        return str(class_value)

    # Change this only if your saved model predicts encoded
    # numeric target values and no label mapping was saved.
    fallback_mapping = {
        0: "High",
        1: "Low",
        2: "Moderate"
    }

    return fallback_mapping.get(
        class_value,
        str(class_value)
    )


def validate_features(
    location_features: Dict[str, Any]
) -> None:
    """
    Validate the features before prediction.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in location_features
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )


def get_classifier_classes() -> list:
    """
    Obtain class order used by predict_proba.
    """

    if hasattr(model_pipeline, "classes_"):
        return list(
            model_pipeline.classes_
        )

    if hasattr(model_pipeline, "named_steps"):

        for step in reversed(
            list(
                model_pipeline.named_steps.values()
            )
        ):
            if hasattr(step, "classes_"):
                return list(step.classes_)

    return []


def predict_feasibility(
    location_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Predict restaurant-location feasibility.
    """

    # Convert raw POI counts (returned by collect_location_features)
    # into the grouped/composite feature set expected by the model.
    def safe_get(name: str) -> float:
        val = location_features.get(name, 0)
        try:
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    grouped = {
        "cultural_activity_score": (
            safe_get("cinema_count_500m")
            + safe_get("museum_count_500m")
            + safe_get("recreation_count_500m")
        ),
        "accessibility_score": (
            safe_get("bus_stop_count_500m") + safe_get("parking_space_count_500m")
        ),
        "competition_pressure_score": safe_get("competitor_count_500m"),
        "education_score": safe_get("college_count_500m") + safe_get("school_count_500m"),
        "market_gap_score": max(
            0.0,
            (safe_get("retail_count_500m") + safe_get("office_count_500m"))
            - safe_get("competitor_count_500m"),
        ),
        "health_activity_score": safe_get("clinic_count_500m") + safe_get("hospital_count_500m"),
        "attraction_score": (
            safe_get("museum_count_500m") + safe_get("cinema_count_500m") + safe_get("temple_count_500m")
        ),
        "commercial_score": (
            safe_get("bank_count_500m") + safe_get("office_count_500m") + safe_get("retail_count_500m")
        ),
        # auxiliary features passed through
        "search_area": location_features.get("search_area"),
        "avg_restaurant_rating_500m": safe_get("avg_restaurant_rating_500m"),
        "avg_review_ratings_500m": safe_get("avg_review_ratings_500m"),
        "nearest_restaurant_m": safe_get("nearest_restaurant_m"),
    }

    validate_features(grouped)

    model_input = pd.DataFrame([grouped], columns=MODEL_FEATURES)

    predicted_class = model_pipeline.predict(
        model_input
    )[0]

    predicted_class = convert_numpy_value(
        predicted_class
    )

    predicted_label = class_to_label(
        predicted_class
    )

    probabilities: Dict[str, float] = {}

    if hasattr(model_pipeline, "predict_proba"):

        probability_values = (
            model_pipeline.predict_proba(
                model_input
            )[0]
        )

        class_values = get_classifier_classes()

        if not class_values:
            class_values = list(
                range(len(probability_values))
            )

        for class_value, probability in zip(
            class_values,
            probability_values
        ):
            class_value = convert_numpy_value(
                class_value
            )

            label = class_to_label(
                class_value
            )

            probabilities[label] = round(
                float(probability),
                6
            )

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probabilities": probabilities
    }


def get_model_information() -> Dict[str, Any]:
    """
    Return model information for the health endpoint.
    """

    return {
        "model_name": model_metadata.get(
            "model_name",
            type(model_pipeline).__name__
        ),
        "model_features": MODEL_FEATURES
    }