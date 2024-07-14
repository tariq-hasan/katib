from typing import List, Optional

from kubernetes.client import V1ObjectMeta

from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1Experiment
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1TrialParameterSpec
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib.constants import constants


def generate_trial_template() -> V1beta1TrialTemplate:
    trial_spec = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "training-container",
                            "image": "docker.io/kubeflowkatib/pytorch-mnist-cpu:v0.14.0",
                            "command": [
                                "python3",
                                "/opt/pytorch-mnist/mnist.py",
                                "--epochs=1",
                                "--batch-size=64",
                                "--lr=${trialParameters.learningRate}",
                                "--momentum=${trialParameters.momentum}",
                            ]
                        }
                    ],
                    "restartPolicy": "Never"
                }
            }
        }
    }

    return V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="learningRate",
                description="Learning rate for the training model",
                reference="lr"
            ),
            V1beta1TrialParameterSpec(
                name="momentum",
                description="Momentum for the training model",
                reference="momentum"
            ),
        ],
        trial_spec=trial_spec
    )


def generate_experiment(
    metadata: V1ObjectMeta,
    algorithm_spec: V1beta1AlgorithmSpec,
    objective_spec: V1beta1ObjectiveSpec,
    parameters: List[V1beta1ParameterSpec],
    trial_template: V1beta1TrialTemplate,
) -> V1beta1Experiment:
    return V1beta1Experiment(
        api_version=constants.API_VERSION,
        kind=constants.EXPERIMENT_KIND,
        metadata=metadata,
        spec=V1beta1ExperimentSpec(
            max_trial_count=3,
            parallel_trial_count=2,
            max_failed_trial_count=1,
            algorithm=algorithm_spec,
            objective=objective_spec,
            parameters=parameters,
            trial_template=trial_template,
        )
    )


def create_experiment(
    name: Optional[str] = None,
    generate_name: Optional[str] = None
) -> V1beta1Experiment:
    experiment_namespace = "test"

    if name is not None:
        metadata = V1ObjectMeta(name=name, namespace=experiment_namespace)
    elif generate_name is not None:
        metadata = V1ObjectMeta(generate_name=generate_name, namespace=experiment_namespace)
    else:
        metadata = V1ObjectMeta(namespace=experiment_namespace)

    algorithm_spec=V1beta1AlgorithmSpec(
        algorithm_name="random"
    )

    objective_spec=V1beta1ObjectiveSpec(
        type="minimize",
        goal= 0.001,
        objective_metric_name="loss",
    )

    parameters=[
        V1beta1ParameterSpec(
            name="lr",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(
                min="0.01",
                max="0.06"
            ),
        ),
        V1beta1ParameterSpec(
            name="momentum",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(
                min="0.5",
                max="0.9"
            ),
        ),
    ]

    trial_template = generate_trial_template()

    experiment = generate_experiment(
        metadata,
        algorithm_spec,
        objective_spec,
        parameters,
        trial_template
    )
    return experiment
