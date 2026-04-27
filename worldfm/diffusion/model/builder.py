from worldfm.diffusion.model.utils import set_grad_checkpoint


class Registry:
    """Small inference-time registry compatible with the local model decorators."""

    def __init__(self, name):
        self.name = name
        self._modules = {}

    def register_module(self, name=None):
        def _wrap(cls):
            self._modules[name or cls.__name__] = cls
            return cls
        return _wrap

    def build(self, cfg, default_args=None):
        cfg = dict(cfg)
        default_args = dict(default_args or {})
        type_name = cfg.pop('type')
        if type_name not in self._modules:
            raise KeyError(f'{type_name!r} is not registered in {self.name}')
        args = {**default_args, **cfg}
        return self._modules[type_name](**args)


MODELS = Registry('models')

_NETS_IMPORTED = False


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    # Ensure model modules are imported and registered before building.
    # Use lazy import to avoid circular-import issues during module init.
    global _NETS_IMPORTED
    if not _NETS_IMPORTED:
        from . import nets  # noqa: F401
        _NETS_IMPORTED = True
    model = MODELS.build(cfg, default_args=kwargs)
    if use_grad_checkpoint:
        set_grad_checkpoint(model, use_fp32_attention=use_fp32_attention, gc_step=gc_step)
    return model
