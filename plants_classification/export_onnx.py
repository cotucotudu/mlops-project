import hydra
import torch
from omegaconf import DictConfig

from plants_classification.train import FlowerResNet50


@hydra.main(version_base=None, config_path="../configs", config_name="export_onnx")
def main(cfg: DictConfig):
    model = FlowerResNet50(
        num_classes=cfg.model.num_classes,
        freeze_backbone=cfg.model.freeze_backbone,
        learning_rate=cfg.model.learning_rate,
    )
    checkpoint = torch.load(cfg.model.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.model.img_size, cfg.model.img_size)
    torch.onnx.export(
        model,
        dummy_input,
        cfg.model.onnx_output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {cfg.model.onnx_output}")


if __name__ == "__main__":
    main()
