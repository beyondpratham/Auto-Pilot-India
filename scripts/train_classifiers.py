import argparse
import os

from autopilot.classical_models import train_and_evaluate_classical_models
from autopilot.cnn_model import train_cnn
from autopilot.config import IMAGE_SIZE, RANDOM_STATE
from autopilot.datasets import load_image_dataset, train_test_split_dataset
from autopilot.metrics import compute_metrics, plot_model_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare classification models")
    parser.add_argument("--data", default="data/real")
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--backbone", default="vgg16", choices=["vgg16", "mobilenet_v2"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def print_results_table(results):
    header = f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<22}{metrics['accuracy']:>10.4f}{metrics['precision']:>11.4f}"
            f"{metrics['recall']:>9.4f}{metrics['f1']:>8.4f}"
        )


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    dataset = load_image_dataset(args.data, tuple(args.image_size))
    x_train, x_test, y_train, y_test = train_test_split_dataset(
        dataset, test_size=args.test_size, random_state=args.seed
    )
    print(f"Loaded {len(dataset.images)} images across classes: {dataset.class_names}")
    print(f"Train: {len(x_train)}  Test: {len(x_test)}")

    results = {}
    models = {}

    classical_models, classical_predictions = train_and_evaluate_classical_models(
        x_train, y_train, x_test, dataset.class_names
    )
    for name, y_pred in classical_predictions.items():
        results[name] = compute_metrics(y_test, y_pred)
        models[name] = classical_models[name]

    print("Training CNN (transfer learning)...")
    cnn_adapter, cnn_pred = train_cnn(
        x_train,
        y_train,
        x_test,
        dataset.class_names,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    results["cnn"] = compute_metrics(y_test, cnn_pred)
    models["cnn"] = cnn_adapter

    print()
    print_results_table(results)

    chart_path = os.path.join(args.output, "model_comparison.png")
    plot_model_comparison(results, chart_path)
    print(f"\nSaved comparison chart to {chart_path}")

    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = models[best_name]
    if best_name == "cnn":
        best_path = os.path.join(args.output, "best_model.pt")
    else:
        best_path = os.path.join(args.output, "best_model.joblib")
    best_model.save(best_path)
    print(f"Best model: {best_name} (F1={results[best_name]['f1']:.4f}) saved to {best_path}")


if __name__ == "__main__":
    main()
