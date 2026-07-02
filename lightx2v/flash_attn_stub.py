# import-safe stub: real flash-attn removed (torch ABI mismatch on this box).
# Every attribute import succeeds and returns a callable that fails LOUDLY on
# use. The wan2.2 t2v distill path (sla/sage attention) never calls flash_attn.
def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "flash_attn stub: %s called but flash-attn is not installed"
            " for torch 2.8+cu128 on this box" % name
        )

    return _unavailable
