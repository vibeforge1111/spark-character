import subprocess
def test_character_version():
    import spark_character
    assert hasattr(spark_character, '__version__')
