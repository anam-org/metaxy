from metaxy.models.field import FieldSpec


def test_default_code_version():
    field = FieldSpec("my_field")

    # this default is EXTREMELY important
    # changing it will affect **all versions on all fields and features**
    assert field.code_version == "__metaxy_initial__"
