from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, TYPE_CHECKING

from pyspark.sql import Row, SparkSession

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torchvision import transforms

DEFAULT_CLASS_NAMES = (
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
)
DEFAULT_IMAGE_SIZE = 224


def build_spark_session(app_name: str = "FruitBatchPipeline") -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.python.worker.connect.timeout", "300")
        .getOrCreate()
    )


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int):
    import torch.nn as nn
    from torchvision import models

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(
    model_path: Path | str,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    device: Any | None = None,
):
    import torch

    device = device or get_device()
    model = build_model(num_classes=len(class_names))
    checkpoint = torch.load(str(model_path), map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        cleaned_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(cleaned_state_dict)
    elif hasattr(checkpoint, "state_dict"):
        model.load_state_dict(checkpoint.state_dict())
    else:
        raise TypeError("El archivo del modelo no contiene un state_dict compatible.")

    model.to(device)
    model.eval()
    return model


def build_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_image(
    content: bytes,
    model,
    transform,
    class_names: Sequence[str],
    device,
) -> tuple[str, float]:
    from PIL import Image
    import torch

    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)

    return class_names[index.item()], float(confidence.item())


def predict_partition(
    rows: Iterable[Row],
    model_path: Path | str,
    class_names: Sequence[str],
    image_size: int,
) -> Iterable[Row]:
    device = get_device()
    model = load_model(model_path=model_path, class_names=class_names, device=device)
    transform = build_transform(image_size=image_size)

    for row in rows:
        path = Path(row.path)
        true_label = path.parent.name
        predicted_label, confidence = predict_image(
            content=row.content,
            model=model,
            transform=transform,
            class_names=class_names,
            device=device,
        )
        yield Row(
            path=str(path),
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
        )


def read_image_paths(image_root: Path | str) -> list[str]:
    root_path = Path(image_root)
    if not root_path.exists():
        raise FileNotFoundError(f"No se encontró la ruta de entrada: {root_path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_paths = sorted(
        file_path
        for file_path in root_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in image_extensions
    )

    if not image_paths:
        raise ValueError(f"No se encontraron imágenes en: {root_path}")

    return [str(file_path) for file_path in image_paths]


def predict_path_partition(
    paths: Iterable[str],
    model_path: Path | str,
    class_names: Sequence[str],
    image_size: int,
) -> Iterable[Row]:
    device = get_device()
    model = load_model(model_path=model_path, class_names=class_names, device=device)
    transform = build_transform(image_size=image_size)

    for path_value in paths:
        path = Path(path_value)
        predicted_label, confidence = predict_image(
            content=path.read_bytes(),
            model=model,
            transform=transform,
            class_names=class_names,
            device=device,
        )
        yield Row(
            path=str(path),
            true_label=path.parent.name,
            predicted_label=predicted_label,
            confidence=confidence,
        )


def predict_path_record(
    path_value: str,
    model,
    transform,
    class_names: Sequence[str],
    device,
) -> Row:
    path = Path(path_value)
    predicted_label, confidence = predict_image(
        content=path.read_bytes(),
        model=model,
        transform=transform,
        class_names=class_names,
        device=device,
    )
    return Row(
        path=str(path),
        true_label=path.parent.name,
        predicted_label=predicted_label,
        confidence=confidence,
    )


def run_batch_pipeline(
    image_root: Path | str,
    model_path: Path | str,
    output_path: Path | str | None = None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    sample_limit: Optional[int] = None,
):
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    device = get_device()
    model = load_model(model_path=model_path, class_names=class_names, device=device)
    transform = build_transform(image_size=image_size)

    image_paths = read_image_paths(image_root)
    if sample_limit is not None:
        sample_size = min(sample_limit, len(image_paths))
        image_paths = random.Random(42).sample(image_paths, sample_size)

    predictions = [
        predict_path_record(
            path_value=path_value,
            model=model,
            transform=transform,
            class_names=tuple(class_names),
            device=device,
        )
        for path_value in image_paths
    ]
    if output_path is not None:
        import pandas as pd

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "predictions.parquet"
        pandas_df = pd.DataFrame([row.asDict() for row in predictions])
        pandas_df.to_parquet(output_file, index=False)

    import pandas as pd

    return pd.DataFrame([row.asDict() for row in predictions])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline batch de inferencia con PySpark.")
    parser.add_argument("--image-root", required=True, help="Ruta con las imágenes de entrada.")
    parser.add_argument("--model-path", required=True, help="Ruta al modelo entrenado (.pth).")
    parser.add_argument("--output-path", required=True, help="Ruta de salida para guardar Parquet.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limita la cantidad de imágenes para una ejecución de prueba.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_df = run_batch_pipeline(
        image_root=args.image_root,
        model_path=args.model_path,
        output_path=args.output_path,
        sample_limit=args.sample_limit,
    )
    result_df.show(20, truncate=False)
    result_df.sparkSession.stop()


if __name__ == "__main__":
    main()
