import os

from model.MPT import get_model as _get_model
from model.MPT import get_loss

from tools.routing_collector import RoutingCollector


_COLLECTOR = None


def get_model(data_config, **kwargs):
    global _COLLECTOR

    model, model_info = _get_model(
        data_config,
        **kwargs,
    )

    output_dir = os.environ.get(
        "ROUTING_OUTPUT_DIR"
    )

    run_name = os.environ.get(
        "ROUTING_RUN_NAME"
    )

    if not output_dir:
        raise RuntimeError(
            "ROUTING_OUTPUT_DIR is required"
        )

    _COLLECTOR = RoutingCollector(
        model=model,
        output_dir=output_dir,
        run_name=run_name,
    )

    return model, model_info
