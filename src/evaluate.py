import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_test_loader
from model import create_resnet18_cifar10
from utils import ensure_dir, get_device, set_seed


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def load_checkpoint(path: str, device: torch.device) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run `python src/train.py` first."
        )
    return torch.load(path, map_location=device, weights_only=False)


def compute_confusion_matrix(y_true, y_pred, num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix


def compute_report(cm: list[list[int]], class_names: list[str]) -> list[dict[str, float | str]]:
    rows = []
    total_support = 0
    total_correct = 0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for class_index, class_name in enumerate(class_names):
        tp = cm[class_index][class_index]
        support = sum(cm[class_index])
        predicted = sum(row[class_index] for row in cm)

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support,
            }
        )

        total_support += support
        total_correct += tp
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    macro_precision = sum(float(row["precision"]) for row in rows) / len(rows)
    macro_recall = sum(float(row["recall"]) for row in rows) / len(rows)
    macro_f1 = sum(float(row["f1-score"]) for row in rows) / len(rows)
    accuracy = total_correct / total_support if total_support else 0.0

    rows.append(
        {
            "class": "accuracy",
            "precision": accuracy,
            "recall": accuracy,
            "f1-score": accuracy,
            "support": total_support,
        }
    )
    rows.append(
        {
            "class": "macro avg",
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": total_support,
        }
    )
    rows.append(
        {
            "class": "weighted avg",
            "precision": weighted_precision / total_support if total_support else 0.0,
            "recall": weighted_recall / total_support if total_support else 0.0,
            "f1-score": weighted_f1 / total_support if total_support else 0.0,
            "support": total_support,
        }
    )
    return rows


def save_classification_report(rows: list[dict[str, float | str]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(cm: list[list[int]], class_names: list[str], output_path: str) -> None:
    cell = 72
    margin_left = 130
    margin_top = 90
    margin_right = 30
    margin_bottom = 120
    width = margin_left + cell * len(class_names) + margin_right
    height = margin_top + cell * len(class_names) + margin_bottom
    max_value = max(max(row) for row in cm) or 1

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((margin_left, 20), "CIFAR-10 ResNet18 Confusion Matrix", fill="black", font=font)
    draw.text((margin_left + cell * 3, height - 35), "Predicted label", fill="black", font=font)
    draw.text((20, margin_top + cell * 4), "True label", fill="black", font=font)

    for i, name in enumerate(class_names):
        x = margin_left + i * cell + 6
        y = margin_top - 22
        draw.text((x, y), name[:8], fill="black", font=font)
        draw.text((12, margin_top + i * cell + 28), name, fill="black", font=font)

    for row_index, row in enumerate(cm):
        for col_index, value in enumerate(row):
            intensity = int(255 - 190 * (value / max_value))
            fill = (intensity, intensity + 20 if intensity <= 235 else 255, 255)
            x0 = margin_left + col_index * cell
            y0 = margin_top + row_index * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(210, 210, 210))
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (x0 + (cell - text_width) / 2, y0 + (cell - text_height) / 2),
                text,
                fill="black",
                font=font,
            )

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 baseline on CIFAR-10.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_resnet18.pth")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-misclassified", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    ensure_dir(args.results_dir)
    misclassified_dir = os.path.join(args.results_dir, "misclassified")
    ensure_dir(misclassified_dir)

    test_loader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_resnet18_cifar10().to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    saved_count = 0
    sample_index = 0
    class_names = list(CIFAR10_CLASSES)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluate"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

            wrong_mask = predictions.ne(labels)
            if wrong_mask.any() and saved_count < args.max_misclassified:
                wrong_indices = wrong_mask.nonzero(as_tuple=False).flatten().cpu()
                restored_images = denormalize(images[wrong_mask]).cpu()
                wrong_labels = labels[wrong_mask].cpu()
                wrong_predictions = predictions[wrong_mask].cpu()

                for batch_index, image, true_label, pred_label in zip(
                    wrong_indices, restored_images, wrong_labels, wrong_predictions
                ):
                    if saved_count >= args.max_misclassified:
                        break
                    dataset_index = sample_index + int(batch_index)
                    filename = (
                        f"idx{dataset_index:05d}_true-{class_names[true_label]}_"
                        f"pred-{class_names[pred_label]}.png"
                    )
                    save_image(image, os.path.join(misclassified_dir, filename))
                    saved_count += 1

            sample_index += images.size(0)

    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))
    report_rows = compute_report(cm, class_names)
    macro_row = next(row for row in report_rows if row["class"] == "macro avg")
    accuracy_row = next(row for row in report_rows if row["class"] == "accuracy")

    print(f"Accuracy:  {float(accuracy_row['f1-score']):.4f}")
    print(f"Precision: {float(macro_row['precision']):.4f}")
    print(f"Recall:    {float(macro_row['recall']):.4f}")
    print(f"F1-score:  {float(macro_row['f1-score']):.4f}")

    report_path = os.path.join(args.results_dir, "classification_report.csv")
    save_classification_report(report_rows, report_path)
    print(f"Saved classification report to {report_path}")

    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    save_confusion_matrix(cm, class_names, cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved {saved_count} misclassified images to {Path(misclassified_dir)}")


if __name__ == "__main__":
    main()
