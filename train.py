from ultralytics import YOLO
import argparse

def train_yolo_model(data_path_yaml, model_path, img_size=1024, epochs=100, batch_size=16):
    """
    Train a YOLO model on the specified dataset.

    Args:
        data_path (str): Path to the dataset configuration file.
        model_path (str): Path to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        None
    """
    # Load a pretrained YOLO11n model
    model = YOLO(model_path)  # Load a pretrained YOLO model

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data=data_path_yaml,  # Path to dataset configuration file
        epochs=epochs,  # Number of training epochs
        imgsz=img_size,  # Image size for training
        batch_size=batch_size,  # Batch size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("test/example1.tiff")  # Predict on an image
    results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model
    print(f"Model exported to: {path}")


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
        "--device",
        default="cuda:0",
        type=str,
        help="Device to run the train on, e.g., 'cuda:0' or 'cpu'.",
    )
    args = parser.parse_args()
    model_path = args.model_path
    data_path_yaml = args.data_path_yaml
    epochs = args.epochs
    img_size = args.img_size
    batch_size = args.batch_size
    device = args.device

    train_yolo_model(
        data_path_yaml=data_path_yaml,
        model_path=model_path,
        img_size=img_size,
        epochs=epochs,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
