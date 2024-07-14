import pytest

from kubeflow.katib.tests.test_data import TEST_RESULT_SUCCESS, test_create_experiment_data


@pytest.mark.parametrize("test_name,kwargs,expected_output", test_create_experiment_data)
def test_create_experiment(katib_client, test_name, kwargs, expected_output):
    """
    test create_experiment function of katib client
    """
    print("\n\nExecuting test:", test_name)
    try:
        katib_client.create_experiment(**kwargs)
        assert expected_output == TEST_RESULT_SUCCESS
    except Exception as e:
        assert type(e) is expected_output
    print("test execution complete")
