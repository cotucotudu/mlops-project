import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

# from plants_classification.data import FlowerDataModule
from plants_classification.train import FlowerResNet50  # importing model from train.py


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    # Loading model from chkpt
    print(cfg)
    checkpoint_path = cfg.infer.model.checkpoint_path
    model = FlowerResNet50.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Same transformations as for val
    preprocess = transforms.Compose(
        [
            transforms.Resize(cfg.infer.preprocess.resize),
            transforms.CenterCrop(cfg.infer.preprocess.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.infer.preprocess.mean, std=cfg.infer.preprocess.std
            ),
        ]
    )

    # Загрузка и предобработка изображения
    img = Image.open(cfg.infer.image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)  # Добавляем batch dimension

    # Перемещение на устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Инференс
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)

    # Получаем топ-k предсказаний
    top_probs, top_idxs = probs.topk(cfg.infer.top_k)

    # Вывод результатов
    print(f"Top-{cfg.infer.top_k} preds for {cfg.infer.image_path}: ")
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        print(f"Class {idx.item()}: probability {prob.item(): .4f}")


if __name__ == "__main__":
    main()
