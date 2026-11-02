from ultralytics import YOLO
import argparse
import torch


def train_yolo_model(
    data_path_yaml,
    model_path,
    img_size=1024,
    epochs=100,
    batch_size=16,
    device="0",
    workers=8,
    amp=True,
    cache=False,
    project=None,
):
    """
    Train a YOLO model on the specified dataset.

    Args:
        data_path_yaml (str): Path to the dataset configuration file.
        model_path (str): Path to the pre-trained model to start from.
        img_size (int): Image size used for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (str): Device to run on (e.g. 'cpu', '0', [0,1,2,3]).
        workers (int): Number of dataloader worker processes.
        amp (bool): Mixed precision training (uses less VRAM, usually faster).
        cache (bool or str): Cache images in RAM ('ram') or on disk ('disk') for faster epochs.

    Returns:
        The trained YOLO model.
    """
    # Load a pretrained YOLO model
    model = YOLO(model_path)  # Load a pretrained YOLO model

    train_args = dict(
        data=data_path_yaml,  # Path to dataset configuration file
        epochs=epochs,  # Number of training epochs
        imgsz=img_size,  # Image size for training
        batch=batch_size,  # Batch size for training
        device=device,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        workers=workers,  # Parallel dataloader workers
        amp=amp,  # Automatic mixed precision
        cache=cache,  # Reuse decoded images across epochs
    )
    if project:
        train_args["project"] = project

    # Train the model
    model.train(**train_args)

    # Evaluate the model's performance on the validation set
    metrics = model.val(device=device)
    print(metrics)

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")
    print(f"Model exported to: {path}")
    return model


def main():
    # Example usage of the trained model
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="training config file path.",
    )
    parser.add_argument("--data_path_yaml", help="datasets config file path.")
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--img_size",
        default=1024,
        type=int,
        help="Image size for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="Number of dataloader worker processes (0 disables multiprocessing).",
    )
    parser.add_argument(
        "--cache",
        default=False,
        action="store_true",
        help="Cache decoded training images to speed up epochs (needs enough RAM).",
    )
    parser.add_argument(
        "--no_amp",
        default=False,
        action="store_true",
        help="Disable mixed precision training.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device to run the train on, e.g., 'cuda:0' or 'cpu'.",
    )
    args = parser.parse_args()
    model_path = args.model_path
    data_path_yaml = args.data_path_yaml
    epochs = args.epochs
    img_size = args.img_size
    batch_size = args.batch_size
    # ultralytics expects the bare index for CUDA devices ('0' instead of 'cuda:0')
    device = args.device.replace("cuda:", "") if args.device.startswith("cuda") else args.device

    train_yolo_model(
        data_path_yaml=data_path_yaml,
        model_path=model_path,
        img_size=img_size,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        workers=args.workers,
        amp=not args.no_amp,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
