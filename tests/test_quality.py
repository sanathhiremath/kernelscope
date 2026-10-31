import pytest
from kernel_scope.quality import agreement


def detection(box=(0., 0., 10., 10.), label=1, score=.9):
    return dict(box=list(box), label=label, score=score)


def test_output_order_does_not_change_agreement():
    left, right = detection(), detection((20., 20., 30., 30.))
    assert agreement([[left, right]], [[right, left]])['passed']


def test_extra_and_missing_detection_fail_gate():
    person = detection()
    assert not agreement([[person]], [[person, person]])['passed']
    assert not agreement([[person]], [[]])['passed']


def test_class_and_geometry_changes_fail_gate():
    assert not agreement([[detection()]], [[detection(label=2)]])['passed']
    assert not agreement([[detection()]], [[detection((1., 1., 11., 11.))]])['passed']


def test_nan_and_empty_outputs_cannot_pass():
    assert not agreement([[detection()]], [[detection(score=float('nan'))]])['passed']
    assert not agreement([[]], [[]])['passed']


def test_mismatched_frames_are_rejected():
    with pytest.raises(ValueError):
        agreement([[]], [])
