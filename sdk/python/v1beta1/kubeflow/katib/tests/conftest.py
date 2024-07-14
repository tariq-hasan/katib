import multiprocessing
from unittest.mock import patch, Mock

import pytest

from kubeflow.katib import KatibClient


class ConflictException(Exception):
    """Exception raised for conflicts.

    This exception is used to indicate that a conflict has occurred,
    typically represented by the HTTP status code 409.
    """
    def __init__(self):
        self.status = 409


def create_namespaced_custom_object_response(*args, **kwargs):
    """Simulates the response of creating a namespaced custom object in a Kubernetes cluster.

    This function is designed to simulate different responses based on the provided arguments,
    primarily used for testing purposes.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        multiprocessing.TimeoutError: If the third argument is "timeout".
        ConflictException: If the third argument is "conflict".
        Exception: If the third argument is "runtime".

    Returns:
        dict: A dictionary with metadata about the created object if the third argument is "test",
              "test-name", or "test-generate-name".

    Examples:
        >>> create_namespaced_custom_object_response(None, None, "test")
        {'metadata': {'name': 'experiment-mnist-ci-test'}}

        >>> create_namespaced_custom_object_response(None, None, "timeout")
        Traceback (most recent call last):
        ...
        multiprocessing.TimeoutError

        >>> create_namespaced_custom_object_response(None, None, "conflict")
        Traceback (most recent call last):
        ...
        ConflictException

        >>> create_namespaced_custom_object_response(None, None, "runtime")
        Traceback (most recent call last):
        ...
        Exception
    """
    if args[2] == "timeout":
        raise multiprocessing.TimeoutError()
    elif args[2] == "conflict":
        raise ConflictException()
    elif args[2] == "runtime":
        raise Exception()
    elif args[2] in ("test", "test-name"):
        return {"metadata": {"name": "experiment-mnist-ci-test"}}
    elif args[2] == "test-generate-name":
        return {"metadata": {"name": "12345-experiment-mnist-ci-test"}}


@pytest.fixture
def katib_client():
    """Fixture for providing a mocked KatibClient instance for testing purposes.

    This fixture patches the Kubernetes CustomObjectsApi and load_kube_config to
    simulate responses for creating namespaced custom objects without needing
    a real Kubernetes cluster. The KatibClient instance returned by this fixture
    can be used in tests to ensure consistent and controlled behavior.

    Yields:
        KatibClient: A mocked instance of KatibClient with patched methods.

    Example:
        def test_example(katib_client):
            # Use the katib_client fixture in your test
            result = katib_client.create_experiment(...)
            assert result == ...
    """
    with patch(
        "kubernetes.client.CustomObjectsApi",
        return_value=Mock(
            create_namespaced_custom_object=Mock(
                side_effect=create_namespaced_custom_object_response
            )
        ),
    ), patch(
        "kubernetes.config.load_kube_config",
        return_value=Mock()
    ):
        client = KatibClient()
        yield client
