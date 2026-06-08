import numpy as np
import pytest
from tabdiff.guidance import parse_constraint_spec, GreaterThan, Mean, Equality, CategoricalConstraint


class _FakeTransform:
    """Identity transform with the sklearn-style .transform(2d) API."""
    def transform(self, row):
        return row


def _fake_info():
    # 2 numeric (age, balance), 1 categorical (education) + a target (income).
    return {
        "column_names": ["age", "balance", "education", "income"],
        "num_col_idx": [0, 1],
        "cat_col_idx": [2],
        "target_col_idx": [3],
        "task_type": "binclass",
        "cat_encoders": {
            "education": ["HS", "Bachelors", "Masters"],
            "income": ["<=50K", ">50K"],
        },
    }


def _parse(spec, **kw):
    info = _fake_info()
    X_num_train = np.zeros((10, 2), dtype=np.float32)
    return parse_constraint_spec(
        spec, info, X_num_train,
        num_transform=_FakeTransform(), int_transform=_FakeTransform(),
        **kw,
    )


def test_parses_numeric_greater_than():
    c = _parse("age>30")
    assert isinstance(c, GreaterThan)
    assert c.idx == 0 and c.target == 30.0


def test_parses_mean_aggregate():
    c = _parse("mean(balance)=1000")
    assert isinstance(c, Mean)
    assert c.idx == 1 and c.target == 1000.0


def test_parses_per_spec_scale_override():
    c = _parse("age>30@scale=0.5")
    assert c.scale == 0.5


def test_parses_categorical_equality_with_target_offset():
    # education is cat_col_idx[0]; task is binclass so col_pos = 0 + 1 = 1.
    c = _parse("education=Bachelors")
    assert isinstance(c, CategoricalConstraint)
    assert c.col_pos == 1 and c.class_idx == 1 and c.sign == 1


def test_parses_target_column_at_position_zero():
    c = _parse("income=>50K")
    assert isinstance(c, CategoricalConstraint)
    assert c.col_pos == 0 and c.class_idx == 1


def test_parses_categorical_not_equal():
    c = _parse("education!=HS")
    assert c.sign == -1 and c.class_idx == 0


def test_numeric_op_on_categorical_raises():
    with pytest.raises(ValueError):
        _parse("education>2")


def test_unknown_column_raises():
    with pytest.raises(ValueError):
        _parse("nope=1")


def _info_target_not_in_encoders():
    # Real TabDiff info.json omits the target column from cat_encoders.
    info = _fake_info()
    del info["cat_encoders"]["income"]
    return info


def test_target_column_missing_from_encoders_raises_without_cat_classes():
    info = _info_target_not_in_encoders()
    X_num_train = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        parse_constraint_spec(
            "income=>50K", info, X_num_train,
            num_transform=_FakeTransform(), int_transform=_FakeTransform(),
        )


def test_target_column_resolves_via_cat_classes_fallback():
    info = _info_target_not_in_encoders()
    X_num_train = np.zeros((10, 2), dtype=np.float32)
    c = parse_constraint_spec(
        "income=>50K", info, X_num_train,
        num_transform=_FakeTransform(), int_transform=_FakeTransform(),
        cat_classes={"income": ["<=50K", ">50K"]},
    )
    assert isinstance(c, CategoricalConstraint)
    assert c.col_pos == 0 and c.class_idx == 1
