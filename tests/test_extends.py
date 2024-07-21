from inspect import signature
from tab_benchmark.utils import extends


def test_basic_extension():
    def base_func(a, b=10, c=20):
        return a + b + c

    @extends(base_func)
    def extended_func(*args, d=10, **kwargs):
        return base_func(*args, **kwargs) + d

    assert extended_func(5, d=15) == 50
    assert str(signature(extended_func)) == '(a, b=10, c=20, *, d=10)'


def test_change_default_value():
    def base_func(a, b=10, c=20):
        return a + b + c

    @extends(base_func, map_default_values_change={'b': 50})
    def extended_func(*args, d=10, **kwargs):
        return base_func(*args, **kwargs) + d

    assert extended_func(5, d=15) == 90
    assert str(signature(extended_func)) == '(a, b=50, c=20, *, d=10)'


def test_exclude_parameter():
    def base_func(a, b=10, c=20):
        return a + b + c

    @extends(base_func, exclude_params=['c'])
    def extended_func(*args, d=10, **kwargs):
        return base_func(*args, **kwargs) + d

    assert extended_func(5, d=15) == 50
    assert str(signature(extended_func)) == '(a, b=10, *, d=10)'


def test_change_default_and_exclude():
    def base_func(a, b=10, c=20):
        return a + b + c

    @extends(base_func, map_default_values_change={'b': 50}, exclude_params=['c'])
    def extended_func(*args, d=10, **kwargs):
        return base_func(*args, **kwargs) + d

    assert extended_func(5, d=15) == 90
    assert str(signature(extended_func)) == '(a, b=50, *, d=10)'
