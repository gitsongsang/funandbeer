# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Lightweight component training function."""
from kfp.v2.dsl import component

@component(
    base_image="python:3.8",
    output_component_file="beer_kfp_train_and_deploy.yaml",
    packages_to_install=["google-cloud-aiplatform"],
)
def train_and_deploy(
    project: str,
    location: str,
    container_uri: str,
    serving_container_uri: str,
    training_file_path: str,
    validation_file_path: str,
    staging_bucket: str,
    regularization: float,
    factors: int,
    iterations: int,
    is_tune: bool,
):

    # pylint: disable-next=import-outside-toplevel
    from google.cloud import aiplatform

    aiplatform.init(
        project=project, location=location, staging_bucket=staging_bucket
    )
    job = aiplatform.CustomContainerTrainingJob(
        display_name="beer_kfp_training",
        container_uri=container_uri,
        command=[
            "python",
            "train.py",
            f"--factors={factors}",
            f"--regularization={regularization}",
            f"--iterations={iterations}",
            f"--is_tune={is_tune}",
        ],
        staging_bucket=staging_bucket,
    )
    model = job.run(replica_count=1, model_display_name="beer_kfp_model")
    endpoint = model.deploy(  # pylint: disable=unused-variable
        traffic_split={"0": 100},
        machine_type="n1-standard-16",
    )