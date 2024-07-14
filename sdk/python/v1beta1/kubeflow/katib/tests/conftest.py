import multiprocessing
from unittest.mock import patch, Mock

import pytest

from kubeflow.katib import KatibClient


class ConflictException(Exception):
    def __init__(self):
        self.status = 409


def create_namespaced_custom_object_response(*args, **kwargs):
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
