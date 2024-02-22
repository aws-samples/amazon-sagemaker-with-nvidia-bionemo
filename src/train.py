# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import boto3
from botocore.exceptions import ClientError
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import timedelta
import yaml

import torch.distributed as dist

NUM_GPUS = int(os.environ.get("SM_NUM_GPUS", 0))
HOSTS = json.loads(os.environ.get("SM_HOSTS", f'["{os.uname()[1]}"]'))
NUM_HOSTS = len(HOSTS)

os.environ["HYDRA_FULL_ERROR"] = "1"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    """Parse the arguments."""
    logging.info("Parsing arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path",
        type=str,
        default="/opt/ml/code",
        help="Path to config files in the container",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="train",
        help="Name of the config file for the run (without file extension)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=[
            "diffdock_confidence",
            "diffdock_score",
            "equidock_db5",
            "equidock_dips",
            "esm1nv",
            "esm2nv_3b",
            "esm2nv_650m",
            "esm2_650m_huggingface",
            "esm2_3b_huggingface",
            "megamolbart",
            "prott5nv",
        ],
        help="Name of BioNeMo model to use for training",
    )

    parser.add_argument(
        "--download-pretrained-weights",
        type=bool,
        default=False,
        help="Download the pre-trained model checkpoint for fine-tuning?",
    )

    parser.add_argument(
        "--ngc-cli-secret-name",
        type=str,
        default="NVIDIA_NGC_CREDS",
        help="Name of an AWS Secrets Manager secret containing NGC_CLI_API_KEY and NGC_CLI_ORG key/value pairs.",
    )

    args, _ = parser.parse_known_args()
    return args


def parse_model_path(
    model_name: str,
    root_path: str = "/workspace/bionemo/examples",
) -> str:
    """Parse the model path from the model name."""

    if model_name == "megamolbart":
        model_path = "molecule/megamolbart/pretrain.py"
    elif model_name == "prott5nv":
        model_path = "protein/prott5nv/pretrain.py"
    elif model_name == "esm1nv":
        model_path = "protein/esm1nv/pretrain.py"
    elif re.match(r"diffdock", model_name):
        model_path = "molecule/diffdock/train.py"
    elif re.match(r"esm2", model_name):
        model_path = "protein/esm2nv/pretrain.py"
    elif re.match(r"equidock", model_name):
        model_path = "protein/equidock/pretrain.py"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return os.path.join(root_path, model_path)


def main(args):
    """Main function."""

    # logging.info(f"Current environment variables are:\n{os.environ}")

    parsed_model_name = args.model_name or get_model_name_from_config(
        args.config_path, args.config_name
    )

    training_script = parse_model_path(parsed_model_name)

    run_cmd = [
        "/usr/bin/python",
        training_script,
        "--config-path",
        args.config_path,
        "--config-name",
        args.config_name,
    ]

    if args.download_pretrained_weights == "True":

        set_ngc_credentials(args.ngc_cli_secret_name)

        logging.info("Downloading pre-trained model checkpoint")
        model_path = os.getenv("MODEL_PATH")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if not os.path.exists("artifact_paths.yaml"):
            shutil.copy(
                "/workspace/bionemo/artifact_paths.yaml",
                os.getcwd(),
            )
        subprocess.run(
            [
                "/usr/bin/python",
                "/workspace/bionemo/download_models.py",
                parsed_model_name,
                "--source",
                "ngc",
                "--download_dir",
                model_path,
            ],
            check=True,
        )
        downloaded_nemo_files = [
            f for f in os.listdir(model_path) if f.endswith(".nemo")
        ]
        checkpoint_path = os.path.join(model_path, downloaded_nemo_files[0])
        logging.info(f"Pre-trained model checkpoint downloaded to {checkpoint_path}")
        run_cmd.append(f"++restore_from_path={checkpoint_path}")

    run_cmd.append(f"++trainer.devices={NUM_GPUS}")
    run_cmd.append(f"++trainer.num_nodes={NUM_HOSTS}")

    logging.info(
        f"Running training script located at {training_script} with command:\n{run_cmd}"
    )

    subprocess.run(
        run_cmd,
        check=True,
    )

    logging.info("Training process complete")

    if os.environ["LOCAL_RANK"] == 0:

        results_path = os.path.join(
            os.getenv("BIONEMO_HOME"), "results/nemo_experiments"
        )
        shutil.copytree(results_path, "/opt/ml/model/")


def set_ngc_credentials(secret_name: str) -> None:
    """Get NVIDIA NGC API Key and org from AWS Secrets Manager"""

    # Create a Secrets Manager client
    client = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION"))

    logging.info("Retrieving NGC credentials from AWS Secrets Manager.")

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    creds = json.loads(get_secret_value_response["SecretString"])

    logging.info("Setting NGC credentials as environment variables.")
    os.environ["NGC_CLI_API_KEY"] = creds.get("NGC_CLI_API_KEY", "")
    os.environ["NGC_CLI_ORG"] = creds.get("NGC_CLI_ORG", "")
    os.environ["NGC_CLI_TEAM"] = creds.get("NGC_CLI_TEAM", "")
    os.environ["NGC_CLI_FORMAT_TYPE"] = creds.get("NGC_CLI_FORMAT_TYPE", "ascii")

    return None


def get_model_name_from_config(
    config_path: str = "/opt/ml/input/data/config", config_name: str = "train"
) -> str:
    """Get the model name from the config file."""
    with open(os.path.join(config_path, f"{config_name}.yaml")) as f:
        config = yaml.safe_load(f)
    return config["name"]


def init_distributed_training(args):
    """Initializes distributed training settings."""

    try:
        backend = "smddp"
        import smdistributed.dataparallel.torch.torch_smddp
    except ModuleNotFoundError:
        backend = "nccl"
        print("Warning: SMDDP not found on this image, falling back to NCCL!")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    if local_rank == 0:
        logging.info("Local Rank is : {}".format(os.environ["LOCAL_RANK"]))
        logging.info("Worldsize is : {}".format(os.environ["WORLD_SIZE"]))
        logging.info("Rank is : {}".format(os.environ["RANK"]))

        logging.info("Master address is : {}".format(os.environ["MASTER_ADDR"]))
        logging.info("Master port is : {}".format(os.environ["MASTER_PORT"]))

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=global_rank,
        init_method="env://",
        timeout=timedelta(seconds=120),
    )

    return local_rank, world_size, global_rank


if __name__ == "__main__":
    args = parse_args()
    local_rank, world_size, global_rank = init_distributed_training(args)
    main(args)
