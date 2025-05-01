import numpy as np

from naviflow_collocated.core.fields import CellField, CellVectorField


def test_cellfield_behavior(subtests):
    n = 10
    field = CellField(n, name="pressure")

    with subtests.test("initialization_zero"):
        assert np.all(field.values == 0.0)

    with subtests.test("set_value"):
        field.set_value(1.23)
        assert np.all(field.values == 1.23)

    with subtests.test("copy_and_mutate"):
        fcopy = field.copy()
        assert fcopy.name == "pressure_copy"
        assert np.allclose(fcopy.values, field.values)
        fcopy.set_value(9.0)
        assert not np.allclose(fcopy.values, field.values)

    with subtests.test("getitem_setitem"):
        field[2] = 99.0
        assert field[2] == 99.0

    with subtests.test("norm"):
        assert np.isclose(field.norm(), np.linalg.norm(field.values))


def test_cellvectorfield_behavior(subtests):
    n = 8
    vec_field = CellVectorField(n, name="velocity")

    with subtests.test("initialization_zero"):
        assert np.all(vec_field.values == 0.0)

    with subtests.test("set_value"):
        vec_field.set_value((1.0, -1.0))
        assert np.all(vec_field.values[:, 0] == 1.0)
        assert np.all(vec_field.values[:, 1] == -1.0)

    with subtests.test("copy_and_mutate"):
        vcopy = vec_field.copy()
        assert vcopy.name == "velocity_copy"
        assert np.allclose(vcopy.values, vec_field.values)
        vcopy.values[:, 1] *= 2.0
        assert not np.allclose(vcopy.values, vec_field.values)

    with subtests.test("getitem_setitem"):
        vec_field[1] = np.array([3.3, 4.4])
        assert np.allclose(vec_field[1], [3.3, 4.4])

    with subtests.test("norm"):
        assert np.isclose(vec_field.norm(), np.linalg.norm(vec_field.values))
