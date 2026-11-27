from paz.inference.serialize.serializable import (
    _register,
    _with_spec,
    build_distribution_fn,
    serializable,
)


def test_serializable_registers_builder():
    @serializable("demo_builder")
    def demo_builder(scale):
        def apply(x):
            return x * scale
        return apply

    fn = build_distribution_fn("demo_builder", scale=3)
    assert fn(2) == 6
    assert fn._paz_spec["fn_id"] == "demo_builder"
    assert fn._paz_spec["kwargs"]["scale"] == 3


def test_register_manual_builder():
    def builder(offset):
        def apply(x):
            return x + offset
        return apply

    _register("manual_builder")(builder)
    fn = build_distribution_fn("manual_builder", offset=5)
    assert fn(1) == 6


def test_with_spec_attaches_metadata():
    def apply(x):
        return x

    wrapped = _with_spec("spec_id", apply, {"value": 1})
    assert wrapped._paz_spec["fn_id"] == "spec_id"
    assert wrapped._paz_spec["kwargs"]["value"] == 1
