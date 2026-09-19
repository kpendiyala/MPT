import os

from model.MPT import get_model as _get_model
from model.MPT import get_loss

from tools.routing_interpretability_collector import RoutingInterpretabilityCollector


_COLLECTOR = None


def get_model(data_config, **kwargs):
    global _COLLECTOR

    model, model_info = _get_model(data_config, **kwargs)

    output_dir = os.environ.get("INTERP_OUTPUT_DIR")
    run_name = os.environ.get("INTERP_RUN_NAME")

    if not output_dir:
        raise RuntimeError("INTERP_OUTPUT_DIR is required")

    _COLLECTOR = RoutingInterpretabilityCollector(
        model=model,
        output_dir=output_dir,
        run_name=run_name,
    )

    return model, model_info
