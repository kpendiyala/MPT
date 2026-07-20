import torch

from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.export_embed = kwargs.pop("export_embed", False)
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        # Standard training/inference path
        if not self.export_embed:
            return self.mod(
                features,
                v=lorentz_vectors,
                mask=mask,
            )

        # Optional embedding-export path
        x, padding_mask = self.mod._forward_encoder(
            features,
            v=lorentz_vectors,
            mask=mask,
        )

        x_cls = self.mod._forward_aggregator(x, padding_mask)

        if self.mod.fc is None:
            return x_cls

        output = self.mod.fc(x_cls)

        if self.mod.for_inference:
            output = torch.softmax(output, dim=1)

        return torch.cat([output, x_cls], dim=1)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts["pf_features"]),
        num_classes=None,

        # Match the high-level MPT architecture
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={
            "dropout": 0,
            "attn_dropout": 0,
            "activation_dropout": 0,
        },
        fc_params=[],
        activation="gelu",

        trim=True,
        for_inference=False,
    )

    cfg.update(**kwargs)
    _logger.info("Dense ParT model config: %s", str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:])
            for k, s in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{
                k: {0: "N", 2: "n_" + k.split("_")[0]}
                for k in data_config.input_names
            },
            "softmax": {0: "N"},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
