# type: ignore
import time
import webbrowser
from pathlib import Path

import pandas as pd
import typer
from filprofiler.api import profile

from saiph import fit
from saiph.lib.size import get_readable_size

N_ROWS = 10000

# Run with:
# poetry run fil-profile python saiph/tests/profile_memory.py True

BASE_PATH = (Path(__file__).parent / "../../").resolve()


def main(sparse: bool) -> None:
    """Profile famd.fit using a fake dataset."""
    df = pd.read_csv(str(BASE_PATH / "tmp/fake_1000000.csv"))

    typer.echo(f"using {get_readable_size(df.memory_usage(index=True).sum())}")

    typer.echo("before fit")
    start = time.perf_counter()
    filename = f"tmp/{time.time()}"
    full_path = BASE_PATH / filename / "index.html"
    profile(lambda: fit(df, nf=5, sparse=sparse), filename)
    end = time.perf_counter()

    typer.echo(f"after fit, took {(end-start):.3} sec")

    webbrowser.open(f"file://{str(full_path)}")


if __name__ == "__main__":
    typer.run(main)
