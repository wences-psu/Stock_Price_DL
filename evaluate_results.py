from src.predict import load_artifacts
from src.preprocessing import load_prepared_data
from src.evaluate import evaluate_model

models = ["lstm", "gru", "cnn"]

for model_name in models:
    for use_volume in [False, True]:
        features = "close_volume" if use_volume else "close"
        folder = f"artifacts/{model_name}_{features}_price-target"

        print(f"\nEvaluating {model_name.upper()} - {features}")

        saved = load_artifacts(folder)

        data = load_prepared_data(
            use_volume=use_volume,
            feature_scaler=saved["feature_scaler"],
            target_scaler=saved["target_scaler"],
        )

        predictions, metrics = evaluate_model(
            saved["model"],
            data,
            f"{model_name.upper()} - {features}",
        )

        print(metrics.to_string())
