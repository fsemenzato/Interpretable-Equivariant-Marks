from importlib import import_module

_MODELS = {
    "MarkSH": "mark",
    "MorphologyMark": "morphology",
    "WebMark": "morphology",
    "HessianMark": "morphology",
}
_ALIASES = {"MarkTIDFourierHWmin": "MorphologyMark"}
REGISTRY = {}


def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls


def get_mark_class(name):
    name = _ALIASES.get(name, name)
    if name not in _MODELS:
        raise ValueError(f"Unknown mark {name!r}; choose from {tuple(_MODELS)}")
    if name not in REGISTRY:
        REGISTRY[name] = getattr(import_module(f"marks.{_MODELS[name]}"), name)
    return REGISTRY[name]


def build_mark(config):
    from types import SimpleNamespace

    values = dict(config) if isinstance(config, dict) else vars(config).copy()
    cls = get_mark_class(values["_model_class"])
    values = {**getattr(cls, "DEFAULTS", {}), **values}
    if values["_model_class"] == "MarkSH":
        values["sh_convention"] = (
            "physics" if values["sh_basis"] == "physical" else "e3nn_component"
        )
    return cls(SimpleNamespace(**values))
