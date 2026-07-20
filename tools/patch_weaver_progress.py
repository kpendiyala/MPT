#!/usr/bin/env python3

import inspect
import re
from pathlib import Path

import weaver.utils.nn.tools as nn_tools


def replace_progress_bar(
    source: str,
    loader_name: str,
    description: str,
) -> tuple[str, bool]:
    desired = (
        f'with tqdm.tqdm('
        f'{loader_name}, '
        f'total=steps_per_epoch, '
        f'desc=f"{description}", '
        f'mininterval=5.0, '
        f'dynamic_ncols=True'
        f') as tq:'
    )

    # Already patched.
    if desired in source:
        return source, False

    possible_patterns = [
        rf"with tqdm\.tqdm\({loader_name}\) as tq:",
        rf"with tqdm\({loader_name}\) as tq:",
    ]

    for pattern in possible_patterns:
        updated, count = re.subn(pattern, desired, source, count=1)
        if count == 1:
            return updated, True

    raise RuntimeError(
        f"Could not locate the tqdm loop for {loader_name}. "
        "The installed Weaver source may use a different format."
    )


def main() -> None:
    tools_path = Path(inspect.getfile(nn_tools))
    source = tools_path.read_text()

    source, train_changed = replace_progress_bar(
        source,
        loader_name="train_loader",
        description="Epoch {epoch} train",
    )

    source, eval_changed = replace_progress_bar(
        source,
        loader_name="test_loader",
        description="Epoch {epoch} validation",
    )

    tools_path.write_text(source)

    print(f"Weaver tools file: {tools_path}")
    print(f"Training progress patched: {train_changed}")
    print(f"Validation progress patched: {eval_changed}")


if __name__ == "__main__":
    main()
