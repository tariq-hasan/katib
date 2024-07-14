from kubeflow.katib.tests.utils import create_experiment

TEST_RESULT_SUCCESS = "success"


test_create_experiment_data = [
    (
        "experiment name and generate_name missing",
        {"experiment": create_experiment()},
        ValueError,
    ),
    (
        "create_namespaced_custom_object timeout error",
        {
            "experiment": create_experiment(name="experiment-mnist-ci-test"),
            "namespace": "timeout",
        },
        TimeoutError,
    ),
    (
        "create_namespaced_custom_object conflict error",
        {
            "experiment": create_experiment(name="experiment-mnist-ci-test"),
            "namespace": "conflict",
        },
        Exception,
    ),
    (
        "create_namespaced_custom_object runtime error",
        {
            "experiment": create_experiment(name="experiment-mnist-ci-test"),
            "namespace": "runtime",
        },
        RuntimeError,
    ),
    (
        "valid flow with experiment type V1beta1Experiment and name",
        {
            "experiment": create_experiment(name="experiment-mnist-ci-test"),
            "namespace": "test-name",
        },
        TEST_RESULT_SUCCESS,
    ),
    (
        "valid flow with experiment type V1beta1Experiment and generate_name",
        {
            "experiment": create_experiment(generate_name="experiment-mnist-ci-test"),
            "namespace": "test-generate-name",
        },
        TEST_RESULT_SUCCESS,
    ),
    (
        "valid flow with experiment JSON and name",
        {
            "experiment": {
                "metadata": {
                    "name": "experiment-mnist-ci-test",
                }
            },
            "namespace": "test-name",
        },
        TEST_RESULT_SUCCESS,
    ),
    (
        "valid flow with experiment JSON and generate_name",
        {
            "experiment": {
                "metadata": {
                    "generate_name": "experiment-mnist-ci-test",
                }
            },
            "namespace": "test-generate-name",
        },
        TEST_RESULT_SUCCESS,
    ),
]
