import argparse
from importlib import import_module
from pathlib import Path
from . import style


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--figures",
        nargs="+",
        default=["tidal", "schematics", "slices", "responses", "constraints", "family"],
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style.FIG = a.out
    style.LOGS = a.out / "logs"
    style.apply()
    for name in a.figures:
        if name not in (
            "tidal",
            "schematics",
            "slices",
            "responses",
            "constraints",
            "family",
        ):
            p.error(f"Unknown figure {name}")
        import_module(f"postprocess.figures.{name}").render(a.data, a.out)


if __name__ == "__main__":
    main()
