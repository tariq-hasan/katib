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
    """Generates a trial template for a Kubernetes Job to run a PyTorch MNIST training job.

    Returns a V1beta1TrialTemplate object configured with a Kubernetes Job spec
    that includes container specifications for PyTorch MNIST training. The container
    uses trial parameters for learning rate and momentum.

    Returns:
        V1beta1TrialTemplate: A trial template for Kubernetes Job spec.

    Example:
        template = generate_trial_template()
        # Use the template to create trials in Katib experiments
    """
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
    """Generates an experiment specification for Katib.

    Args:
        metadata (V1ObjectMeta): Metadata for the experiment.
        algorithm_spec (V1beta1AlgorithmSpec): Algorithm specification for the experiment.
        objective_spec (V1beta1ObjectiveSpec): Objective specification for the experiment.
        parameters (List[V1beta1ParameterSpec]): List of parameter specifications for trials.
        trial_template (V1beta1TrialTemplate): Template for trials in the experiment.

    Returns:
        V1beta1Experiment: A configured experiment object ready for Katib.

    Example:
        metadata = V1ObjectMeta(name="my-experiment")
        algorithm_spec = V1beta1AlgorithmSpec(algorithm_name="random")
        objective_spec = V1beta1ObjectiveSpec(type="minimize", goal=0.001, objective_metric_name="loss")
        parameters = [
            V1beta1ParameterSpec(name="lr", parameter_type="double", feasible_space=V1beta1FeasibleSpace(min="0.01", max="0.06")),
            V1beta1ParameterSpec(name="momentum", parameter_type="double", feasible_space=V1beta1FeasibleSpace(min="0.5", max="0.9")),
        ]
        trial_template = generate_trial_template()

        experiment = generate_experiment(metadata, algorithm_spec, objective_spec, parameters, trial_template)
        # Use experiment to create a Katib experiment in Kubernetes
    """
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
    """Creates an experiment configuration for Katib.

    Args:
        name (Optional[str], optional): Name of the experiment. Defaults to None.
        generate_name (Optional[str], optional): Generated name of the experiment. Defaults to None.

    Returns:
        V1beta1Experiment: Configured experiment object ready for creation.

    Raises:
        ValueError: If both `name` and `generate_name` are None.

    Example:
        # Create an experiment with a specific name
        experiment = create_experiment(name="my-experiment")

        # Alternatively, create an experiment with a generated name
        experiment = create_experiment(generate_name="experiment-12345")

        # Or create an experiment with default settings (namespace "test")
        experiment = create_experiment()

        # Use `experiment` to create a Katib experiment in Kubernetes
    """
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


def train_mnist_model(parameters):
    """Trains a convolutional neural network (CNN) model on the MNIST dataset.

    Args:
        parameters (dict): A dictionary containing the following hyperparameters:
            - lr (float): Learning rate for the optimizer.
            - num_epoch (int): Number of epochs to train the model.
            - is_dist (bool): Flag indicating if distributed training should be used.
            - num_workers (int): Number of workers in distributed training.

    Returns:
        None

    Raises:
        ValueError: If any required parameter is missing or invalid.

    Example:
        # Example parameters
        parameters = {
            "lr": 0.01,
            "num_epoch": 10,
            "is_dist": False,
            "num_workers": 1
        }
        train_mnist_model(parameters)
    """
    import tensorflow as tf
    import numpy as np
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.INFO,
    )
    logging.info("--------------------------------------------------------------------------------------")
    logging.info(f"Input Parameters: {parameters}")
    logging.info("--------------------------------------------------------------------------------------\n\n")


    # Get HyperParameters from the input params dict.
    lr = float(parameters["lr"])
    num_epoch = int(parameters["num_epoch"])

    # Set dist parameters and strategy.
    is_dist = parameters["is_dist"]
    num_workers = parameters["num_workers"]
    batch_size_per_worker = 64
    batch_size_global = batch_size_per_worker * num_workers
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CollectiveCommunication.RING
        )
    )

    # Callback class for logging training.
    # Katib parses metrics in this format: <metric-name>=<metric-value>.
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logging.info(
                "Epoch {}/{}. accuracy={:.4f} - loss={:.4f}".format(
                    epoch+1, num_epoch, logs["accuracy"], logs["loss"]
                )
            )

    # Prepare MNIST Dataset.
    def mnist_dataset(batch_size):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train / np.float32(255)
        y_train = y_train.astype(np.int64)
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(60000)
            .repeat()
            .batch(batch_size)
        )
        return train_dataset

    # Build and compile CNN Model.
    def build_and_compile_cnn_model():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            metrics=["accuracy"],
        )
        return model
    
    # Download Dataset.
    dataset = mnist_dataset(batch_size_global)

    # For dist strategy we should build model under scope().
    if is_dist:
        logging.info("Running Distributed Training")
        logging.info("--------------------------------------------------------------------------------------\n\n")
        with strategy.scope():
            model = build_and_compile_cnn_model()
    else:
        logging.info("Running Single Worker Training")
        logging.info("--------------------------------------------------------------------------------------\n\n")
        model = build_and_compile_cnn_model()
    
    # Start Training.
    model.fit(
        dataset,
        epochs=num_epoch,
        steps_per_epoch=70,
        callbacks=[CustomCallback()],
        verbose=0,
    )
