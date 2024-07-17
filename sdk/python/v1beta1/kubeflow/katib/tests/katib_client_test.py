import pytest

from kubeflow.katib.tests.test_data import TEST_RESULT_SUCCESS, test_create_experiment_data, test_tune_data


@pytest.mark.parametrize("test_name,kwargs,expected_output", test_create_experiment_data)
def test_create_experiment(katib_client, test_name, kwargs, expected_output):
    """Test function for `create_experiment` method of `katib_client`.

    Parameters:
    - katib_client (fixture): Fixture providing an instance of `KatibClient`.
    - test_name (str): Name of the test case.
    - kwargs (dict): Keyword arguments to pass to `create_experiment`.
    - expected_output (type): Expected exception type or `TEST_RESULT_SUCCESS`.

    Raises:
    - AssertionError: If the test fails to assert the expected behavior.

    Returns:
    - None
    """
    print("\n\nExecuting test:", test_name)
    try:
        katib_client.create_experiment(**kwargs)
        assert expected_output == TEST_RESULT_SUCCESS
    except Exception as e:
        assert type(e) is expected_output
    print("test execution complete")


@pytest.mark.parametrize("test_name,kwargs,expected_output", test_tune_data)
def test_tune(katib_client, test_name, kwargs, expected_output):
    """Test function for `tune` method of `katib_client`.

    Parameters:
    - katib_client (fixture): Fixture providing an instance of `KatibClient`.
    - test_name (str): Name of the test case.
    - kwargs (dict): Keyword arguments to pass to `tune`.
    - expected_output (type): Expected exception type or `TEST_RESULT_SUCCESS`.

    Raises:
    - AssertionError: If the test fails to assert the expected behavior.

    Returns:
    - None
    """
    print("\n\nExecuting test:", test_name)
    try:
        katib_client.tune(**kwargs)
        assert expected_output == TEST_RESULT_SUCCESS
    except Exception as e:
        assert type(e) is expected_output
    print("Test execution complete")
