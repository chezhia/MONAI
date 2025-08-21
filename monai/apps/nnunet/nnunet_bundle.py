# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union, Tuple

import numpy as np
import torch
from torch.backends import cudnn

from monai.data.meta_tensor import MetaTensor
from monai.utils import optional_import

join, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="join")
load_json, _ = optional_import("batchgenerators.utilities.file_and_folder_operations", name="load_json")

__all__ = [
    "get_nnunet_trainer",
    "get_nnunet_monai_predictor",
    "get_network_from_nnunet_plans",
    "convert_nnunet_to_monai_bundle",
    "convert_monai_bundle_to_nnunet",
    "ModelnnUNetWrapper",
    "EnsembleProbabilitiesToSegmentation"
]



# Convert a single nnUNet model checkpoint to MONAI bundle format
# The function saves the converted model checkpoint and configuration files in the specified bundle root folder.
def convert_nnunet_to_monai_bundle(nnunet_config: dict, bundle_root_folder: str, fold: int = 0) -> None:
    """
    Convert nnUNet model checkpoints and configuration to MONAI bundle format.

    Parameters
    ----------
    nnunet_config : dict
        Configuration dictionary for nnUNet, containing keys such as 'dataset_name_or_id', 'nnunet_configuration',
        'nnunet_trainer', and 'nnunet_plans'.
    bundle_root_folder : str
        Root folder where the MONAI bundle will be saved.
    fold : int, optional
        Fold number of the nnUNet model to be converted, by default 0.

    Returns
    -------
    None
    """

    nnunet_trainer = "nnUNetTrainer"
    nnunet_plans = "nnUNetPlans"
    nnunet_configuration = "3d_fullres"

    if "nnunet_trainer" in nnunet_config:
        nnunet_trainer = nnunet_config["nnunet_trainer"]

    if "nnunet_plans" in nnunet_config:
        nnunet_plans = nnunet_config["nnunet_plans"]

    if "nnunet_configuration" in nnunet_config:
        nnunet_configuration = nnunet_config["nnunet_configuration"]

    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(nnunet_config["dataset_name_or_id"])
    nnunet_model_folder = Path(os.environ["nnUNet_results"]).joinpath(
        dataset_name, f"{nnunet_trainer}__{nnunet_plans}__{nnunet_configuration}"
    )

    nnunet_checkpoint_final = torch.load(Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_final.pth"), weights_only=False)
    nnunet_checkpoint_best = torch.load(Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_best.pth"), weights_only=False)

    nnunet_checkpoint = {}
    nnunet_checkpoint["inference_allowed_mirroring_axes"] = nnunet_checkpoint_final["inference_allowed_mirroring_axes"]
    nnunet_checkpoint["init_args"] = nnunet_checkpoint_final["init_args"]
    nnunet_checkpoint["trainer_name"] = nnunet_checkpoint_final["trainer_name"]

    print('Creating model checkpoints at: ', Path(bundle_root_folder).joinpath("models", nnunet_configuration))
    os.makedirs(Path(bundle_root_folder).joinpath("models", nnunet_configuration), exist_ok=True)
    
    print(' Saving nnunet_checkpoint.pth at: ', Path(bundle_root_folder).joinpath("models", nnunet_configuration, "nnunet_checkpoint.pth"))
    torch.save(nnunet_checkpoint, Path(bundle_root_folder).joinpath("models", nnunet_configuration, "nnunet_checkpoint.pth"))

    Path(bundle_root_folder).joinpath("models", nnunet_configuration, f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    monai_last_checkpoint = {}
    monai_last_checkpoint["network_weights"] = nnunet_checkpoint_final["network_weights"]
    torch.save(monai_last_checkpoint, Path(bundle_root_folder).joinpath("models", nnunet_configuration, f"fold_{fold}", "model.pt"))

    monai_best_checkpoint = {}
    monai_best_checkpoint["network_weights"] = nnunet_checkpoint_best["network_weights"]
    torch.save(monai_best_checkpoint, Path(bundle_root_folder).joinpath("models", nnunet_configuration, f"fold_{fold}", "best_model.pt"))

    if not os.path.exists(os.path.join(bundle_root_folder, "models", "plans.json")):
        shutil.copy(
            Path(nnunet_model_folder).joinpath("plans.json"), Path(bundle_root_folder).joinpath("models", "plans.json")
        )

    if not os.path.exists(os.path.join(bundle_root_folder, "models", "dataset.json")):
        shutil.copy(
            Path(nnunet_model_folder).joinpath("dataset.json"),
            Path(bundle_root_folder).joinpath("models", "dataset.json"),
        )


# A function to convert all nnunet models (configs and folds) to MONAI bundle format.
# The function iterates through all folds and configurations, converting each model to the specified bundle format.
# The number of folds, configurations, plans and dataset.json will be parsed from the nnunet folder
def convert_best_nnunet_to_monai_bundle(nnunet_config: dict, bundle_root_folder: str, inference_info_file: str = "inference_information.json") -> None:
    """
    Convert all nnUNet models (configs and folds) to MONAI bundle format.

    Parameters
    ----------
    nnunet_config : dict
        Configuration dictionary for nnUNet. Expected keys are:
        - "dataset_name_or_id": str, name or ID of the dataset.
        - "nnunet_configuration": str, configuration name.
        - "nnunet_trainer": str, optional, name of the nnU-Net trainer (default is "nnUNetTrainer").
        - "nnunet_plans": str, optional, name of the nnU-Net plans (default is "nnUNetPlans").
    bundle_root_folder : str
        Path to the root folder of the MONAI bundle.
    inference_info : str, optional
        Path to the inference information file (default is "inference_information.json").

    Returns
    -------
    None
    """
    from batchgenerators.utilities.file_and_folder_operations import subfiles

    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    
    dataset_name = nnunet_config["dataset_name_or_id"]
    
    inference_info_path = Path(os.environ["nnUNet_results"]).joinpath(maybe_convert_to_dataset_name(dataset_name), inference_info_file)
    
    if not os.path.exists(inference_info_path):
        raise FileNotFoundError(f"Inference information file not found: {inference_info_path}")
    inference_info = load_json(inference_info_path)
    
    # Get the best model or ensemble from the inference information
    if "best_model_or_ensemble" not in inference_info:
        raise KeyError(f"Key 'best_model_or_ensemble' not found in inference information file: {inference_info_path}")
    best_model_dict = inference_info["best_model_or_ensemble"]
    
    # Get the folds information
    if "folds" not in inference_info:
        raise KeyError(f"Key 'folds' not found in inference information file: {inference_info_path}")
    folds = inference_info["folds"] # list of folds
    
    cascade_3d_fullres = False
    for model_dict in best_model_dict['selected_model_or_models']:
        if model_dict["configuration"] == "3d_cascade_fullres":
            cascade_3d_fullres = True
        
        print('Converting model: ', model_dict["configuration"])
        nnunet_model_folder = Path(os.environ["nnUNet_results"]).joinpath(
        maybe_convert_to_dataset_name(dataset_name),
        f"{model_dict['trainer']}__{model_dict['plans_identifier']}__{model_dict['configuration']}"
        )
        nnunet_config["nnunet_configuration"] = model_dict["configuration"]
        nnunet_config["nnunet_trainer"] = model_dict["trainer"]
        nnunet_config["nnunet_plans"] = model_dict["plans_identifier"]
        
        if not os.path.exists(nnunet_model_folder):
            raise FileNotFoundError(f"Model folder not found: {nnunet_model_folder}")
        
        for fold in folds:
            print('Converting fold: ', fold, ' of model: ', model_dict["configuration"])
            convert_nnunet_to_monai_bundle_v2(nnunet_config, bundle_root_folder, fold)
    
    # IF model is a cascade model, 3d_lowres is also needed
    if cascade_3d_fullres:
        # check if 3d_lowres is already in the bundle
        if not os.path.exists(os.path.join(bundle_root_folder, "models", "3d_lowres")):
            # copy the 3d_lowres model folder from nnunet results
            nnunet_model_folder = Path(os.environ["nnUNet_results"]).joinpath(
                maybe_convert_to_dataset_name(dataset_name),
                f"{model_dict['trainer']}__{model_dict['plans_identifier']}__3d_lowres"
            )
            if not os.path.exists(nnunet_model_folder):
                raise FileNotFoundError(f"Model folder not found: {nnunet_model_folder}")
            # copy the 3d_lowres model folder to the bundle root folder
            nnunet_config["nnunet_configuration"] = "3d_lowres"
            nnunet_config["nnunet_trainer"] = best_model_dict['selected_model_or_models'][-1]["trainer"] # Using the same trainer as the cascade model
            nnunet_config["nnunet_plans"] = best_model_dict['selected_model_or_models'][-1]["plans_identifier"]  # Using the same plans id as the cascade model
            print('Converting model: ', "3d_lowres")
            for fold in folds:
                print('Converting fold: ', fold, ' of model: ', "3d_lowres")
                convert_nnunet_to_monai_bundle_v2(nnunet_config, bundle_root_folder, fold)
    
    # Finally if postprocessing is needed (for ensemble models)
    if "postprocessing_file" in best_model_dict:
        postprocessing_file_path = best_model_dict["postprocessing_file"]
        if not os.path.exists(postprocessing_file_path):
            raise FileNotFoundError(f"Postprocessing file not found: {postprocessing_file_path}")
        shutil.copy(postprocessing_file_path, Path(bundle_root_folder).joinpath("models", "postprocessing.pkl"))

def convert_monai_bundle_to_nnunet(nnunet_config: dict, bundle_root_folder: str, fold: int = 0) -> None:
    """
    Convert a MONAI bundle to nnU-Net format.

    Parameters
    ----------
    nnunet_config : dict
        Configuration dictionary for nnU-Net. Expected keys are:
        - "dataset_name_or_id": str, name or ID of the dataset.
        - "nnunet_trainer": str, optional, name of the nnU-Net trainer (default is "nnUNetTrainer").
        - "nnunet_plans": str, optional, name of the nnU-Net plans (default is "nnUNetPlans").
    bundle_root_folder : str
        Path to the root folder of the MONAI bundle.
    fold : int, optional
        Fold number for cross-validation (default is 0).

    Returns
    -------
    None
    """
    from odict import odict

    nnunet_trainer: str = "nnUNetTrainer"
    nnunet_plans: str = "nnUNetPlans"

    if "nnunet_trainer" in nnunet_config:
        nnunet_trainer = nnunet_config["nnunet_trainer"]

    if "nnunet_plans" in nnunet_config:
        nnunet_plans = nnunet_config["nnunet_plans"]

    from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    def subfiles(
        folder: Union[str, Path], prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True
    ) -> list[str]:
        res = [
            i.name
            for i in Path(folder).iterdir()
            if i.is_file()
            and (prefix is None or i.name.startswith(prefix))
            and (suffix is None or i.name.endswith(suffix))
        ]
        if sort:
            res.sort()
        return res

    nnunet_model_folder: Path = Path(os.environ["nnUNet_results"]).joinpath(
        maybe_convert_to_dataset_name(nnunet_config["dataset_name_or_id"]),
        f"{nnunet_trainer}__{nnunet_plans}__3d_fullres",
    )

    nnunet_preprocess_model_folder: Path = Path(os.environ["nnUNet_preprocessed"]).joinpath(
        maybe_convert_to_dataset_name(nnunet_config["dataset_name_or_id"])
    )

    Path(nnunet_model_folder).joinpath(f"fold_{fold}").mkdir(parents=True, exist_ok=True)

    nnunet_checkpoint: dict = torch.load(f"{bundle_root_folder}/models/nnunet_checkpoint.pth",weights_only=False)
    latest_checkpoints: list[str] = subfiles(
        Path(bundle_root_folder).joinpath("models", f"fold_{fold}"), prefix="checkpoint_epoch", sort=True
    )
    epochs: list[int] = []
    for latest_checkpoint in latest_checkpoints:
        epochs.append(int(latest_checkpoint[len("checkpoint_epoch=") : -len(".pt")]))

    epochs.sort()
    final_epoch: int = epochs[-1]
    monai_last_checkpoint: dict = torch.load(
        f"{bundle_root_folder}/models/fold_{fold}/checkpoint_epoch={final_epoch}.pt", weights_only=False
    )

    best_checkpoints: list[str] = subfiles(
        Path(bundle_root_folder).joinpath("models", f"fold_{fold}"), prefix="checkpoint_key_metric", sort=True
    )
    key_metrics: list[str] = []
    for best_checkpoint in best_checkpoints:
        key_metrics.append(str(best_checkpoint[len("checkpoint_key_metric=") : -len(".pt")]))

    key_metrics.sort()
    best_key_metric: str = key_metrics[-1]
    monai_best_checkpoint: dict = torch.load(
        f"{bundle_root_folder}/models/fold_{fold}/checkpoint_key_metric={best_key_metric}.pt", weights_only=False
    )

    if "optimizer_state" in monai_last_checkpoint:
        nnunet_checkpoint["optimizer_state"] = monai_last_checkpoint["optimizer_state"]

    nnunet_checkpoint["network_weights"] = odict()

    for key in monai_last_checkpoint["network_weights"]:
        nnunet_checkpoint["network_weights"][key] = monai_last_checkpoint["network_weights"][key]

    nnunet_checkpoint["current_epoch"] = final_epoch
    nnunet_checkpoint["logging"] = nnUNetLogger().get_checkpoint()
    nnunet_checkpoint["_best_ema"] = 0
    nnunet_checkpoint["grad_scaler_state"] = None

    torch.save(nnunet_checkpoint, Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_final.pth"))

    nnunet_checkpoint["network_weights"] = odict()

    if "optimizer_state" in monai_last_checkpoint:
        nnunet_checkpoint["optimizer_state"] = monai_best_checkpoint["optimizer_state"]

    for key in monai_best_checkpoint["network_weights"]:
        nnunet_checkpoint["network_weights"][key] = monai_best_checkpoint["network_weights"][key]

    torch.save(nnunet_checkpoint, Path(nnunet_model_folder).joinpath(f"fold_{fold}", "checkpoint_best.pth"))

    if not os.path.exists(os.path.join(nnunet_model_folder, "dataset.json")):
        shutil.copy(f"{bundle_root_folder}/models/dataset.json", nnunet_model_folder)
    if not os.path.exists(os.path.join(nnunet_model_folder, "plans.json")):
        shutil.copy(f"{bundle_root_folder}/models/plans.json", nnunet_model_folder)
    if not os.path.exists(os.path.join(nnunet_model_folder, "dataset_fingerprint.json")):
        shutil.copy(f"{nnunet_preprocess_model_folder}/dataset_fingerprint.json", nnunet_model_folder)
    if not os.path.exists(os.path.join(nnunet_model_folder, "nnunet_checkpoint.pth")):
        shutil.copy(f"{bundle_root_folder}/models/nnunet_checkpoint.pth", nnunet_model_folder)

# This function loads a nnUNet network from the provided plans and dataset files.
# It initializes the network architecture and loads the model weights if a checkpoint is provided.
def get_network_from_nnunet_plans(
    plans_file: str,
    dataset_file: str,
    configuration: str,
    model_ckpt: Optional[str] = None,
    model_key_in_ckpt: str = "model",
) -> Union[torch.nn.Module, Any]:
    """
    Load and initialize a nnUNet network based on nnUNet plans and configuration.

    Parameters
    ----------
    plans_file : str
        Path to the JSON file containing the nnUNet plans.
    dataset_file : str
        Path to the JSON file containing the dataset information.
    configuration : str
        The configuration name to be used from the plans.
    model_ckpt : Optional[str], optional
        Path to the model checkpoint file. If None, the network is returned without loading weights (default is None).
    model_key_in_ckpt : str, optional
        The key in the checkpoint file that contains the model state dictionary (default is "model").

    Returns
    -------
    network : torch.nn.Module
        The initialized neural network, with weights loaded if `model_ckpt` is provided.
    """
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = load_json(plans_file)
    dataset_json = load_json(dataset_file)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)

    enable_deep_supervision = True

    network = get_network_from_plans(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        allow_init=True,
        deep_supervision=enable_deep_supervision,
    )

    if model_ckpt is None:
        return network
    else:
        state_dict = torch.load(model_ckpt, weights_only=False)
        network.load_state_dict(state_dict[model_key_in_ckpt])
        return network

def get_nnunet_trainer(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    use_compressed_data: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    device: str = "cuda",
    pretrained_model: Optional[str] = None,
) -> Any:  # type: ignore
    """
    Get the nnUNet trainer instance based on the provided configuration.
    The returned nnUNet trainer can be used to initialize the SupervisedTrainer for training, including the network,
    optimizer, loss function, DataLoader, etc.

    Example::

        from monai.apps import SupervisedTrainer
        from monai.bundle.nnunet import get_nnunet_trainer

        dataset_name_or_id = 'Task009_Spleen'
        fold = 0
        configuration = '3d_fullres'
        nnunet_trainer = get_nnunet_trainer(dataset_name_or_id, configuration, fold)

        trainer = SupervisedTrainer(
            device=nnunet_trainer.device,
            max_epochs=nnunet_trainer.num_epochs,
            train_data_loader=nnunet_trainer.dataloader_train,
            network=nnunet_trainer.network,
            optimizer=nnunet_trainer.optimizer,
            loss_function=nnunet_trainer.loss_function,
            epoch_length=nnunet_trainer.num_iterations_per_epoch,
        )

    Parameters
    ----------
    dataset_name_or_id : Union[str, int]
        The name or ID of the dataset to be used.
    configuration : str
        The configuration name for the training.
    fold : Union[int, str]
        The fold number or 'all' for cross-validation.
    trainer_class_name : str, optional
        The class name of the trainer to be used. Default is 'nnUNetTrainer'.
        For a complete list of supported trainers, check:
        https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants
    plans_identifier : str, optional
        Identifier for the plans to be used. Default is 'nnUNetPlans'.
    use_compressed_data : bool, optional
        Whether to use compressed data. Default is False.
    continue_training : bool, optional
        Whether to continue training from a checkpoint. Default is False.
    only_run_validation : bool, optional
        Whether to only run validation. Default is False.
    disable_checkpointing : bool, optional
        Whether to disable checkpointing. Default is False.
    device : str, optional
        The device to be used for training. Default is 'cuda'.
    pretrained_model : Optional[str], optional
        Path to the pretrained model file.

    Returns
    -------
    nnunet_trainer : object
        The nnUNet trainer instance.
    """
    # From nnUNet/nnunetv2/run/run_training.py#run_training
    if isinstance(fold, str):
        if fold != "all":
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!'
                )
                raise e

    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    nnunet_trainer = get_trainer_from_args(
        str(dataset_name_or_id),
        configuration,
        fold,
        trainer_class_name,
        plans_identifier,
        device=torch.device(device),
    )
    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (continue_training and only_run_validation), "Cannot set --c and --val flag at the same time. Dummy."

    maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation)
    nnunet_trainer.on_train_start()  # Added to Initialize Trainer
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if pretrained_model is not None:
        state_dict = torch.load(pretrained_model, weights_only=False)
        if "network_weights" in state_dict:
            nnunet_trainer.network._orig_mod.load_state_dict(state_dict["network_weights"])
    return nnunet_trainer


def get_nnunet_monai_predictor(model_folder: Union[str, Path], model_name: str = "model.pt", dataset_json: dict = None, plans: dict = None, nnunet_config: dict = None,
                               save_probabilities: bool = False, save_files: bool = False, use_folds: Optional[Union[int, str]] = None) -> ModelnnUNetWrapper:   
    """
    Initializes and returns a `nnUNetMONAIModelWrapper` containing the corresponding `nnUNetPredictor`.
    The model folder should contain the following files, created during training:

        - dataset.json: from the nnUNet results folder
        - plans.json: from the nnUNet results folder
        - nnunet_checkpoint.pth: The nnUNet checkpoint file, containing the nnUNet training configuration
        - model.pt: The checkpoint file containing the model weights.

    The returned wrapper object can be used for inference with MONAI framework:
    Example::

        from monai.bundle.nnunet import get_nnunet_monai_predictor

        model_folder = 'path/to/monai_bundle/model'
        model_name = 'model.pt'
        wrapper = get_nnunet_monai_predictor(model_folder, model_name)

        # Perform inference
        input_data = ...
        output = wrapper(input_data)


    Parameters
    ----------
    model_folder : Union[str, Path]
        The folder where the model is stored.
    model_name : str, optional
        The name of the model file, by default "model.pt".
    dataset_json : dict, optional
        The dataset JSON file containing dataset information.
    plans : dict, optional
        The plans JSON file containing model configuration.
    nnunet_config : dict, optional
        The nnUNet configuration dictionary containing model parameters.

    Returns
    -------
    ModelnnUNetWrapper
        A wrapper object that contains the nnUNetPredictor and the loaded model.
    """

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device("cuda", 0),
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    # initializes the network architecture, loads the checkpoint
    print('nnunet_predictor: Model Folder: ', model_folder)
    print('nnunet_predictor: Model name: ', model_name)
    print('nnunet_predictor: use_folds: ', use_folds)
    wrapper = ModelnnUNetWrapper(predictor, 
                                 model_folder = model_folder, 
                                 checkpoint_name = model_name, 
                                 dataset_json = dataset_json, 
                                 plans = plans, 
                                 nnunet_config = nnunet_config, 
                                 save_probabilities = save_probabilities, 
                                 save_files = save_files, 
                                 use_folds = use_folds)
    return wrapper

def get_nnunet_monai_predictors_for_ensemble(
    model_list: list, 
    model_path: Union[str, Path], 
    model_name: str = "model.pt",
    use_folds: Optional[Union[int, str]] = None
) -> Tuple[ModelnnUNetWrapper, ...]:   
    network_list = []
    for model_config in model_list:
        model_folder = Path(model_path).joinpath(model_config)
        print('Model folder: ', model_folder)
        print('Model name: ', model_name)
        print('use_folds: ', use_folds)
        network_list.append(get_nnunet_monai_predictor(model_folder = model_folder, model_name = model_name, save_probabilities=True, save_files=True, use_folds=use_folds))
    return tuple(network_list)


from monai.transforms import MapTransform
from monai.data.meta_tensor import MetaTensor
from monai.config import KeysCollection
from nnunetv2.ensembling.ensemble import average_probabilities
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from typing import List, Dict, Union
import numpy as np
import os


class EnsembleProbabilitiesToSegmentation(MapTransform):
    """
    MONAI transform that loads .npz probability files from metadata['saved_file'] for a given key,
    averages them, and converts to final segmentation using nnU-Net's LabelManager.
    Returns a MetaTensor segmentation result (instead of saving to disk).
    """

    def __init__(
        self,
        keys: KeysCollection,
        dataset_json_path: str,
        plans_json_path: str,
        allow_missing_keys: bool = False,
        output_key: str = "pred",
    ):
        super().__init__(keys, allow_missing_keys)

        # Load required nnU-Net configs
        self.plans_manager = PlansManager(plans_json_path)
        self.dataset_json = self._load_json(dataset_json_path)
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        self.output_key = output_key
        
    def _load_json(self, path: str) -> Dict:
        import json
        with open(path, "r") as f:
            return json.load(f)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        print('Data keys: ', d.keys())
        all_files = []
        for key in self.keys:
            print('Key: ', key)
            #print('Meta keys: ', d[key].meta.keys())
            meta = d[key].meta if isinstance(d[key], MetaTensor) else d.get("meta", {})
            dmeta = dict(meta)
            print("Meta keys: ", dmeta.keys())
            saved_file = meta.get("saved_file", None)

            # Support multiple files for ensemble
            if isinstance(saved_file, str):
                saved_file = [saved_file]
            elif not isinstance(saved_file, list):
                raise ValueError(f"'saved_file' in meta must be str or List[str], got {type(saved_file)}")

            for f in saved_file:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Probability file not found: {f}")
                all_files.append(f)

        print('All files to average: ', all_files)
        # Step 1: average probabilities
        avg_probs = average_probabilities(all_files)

        # Step 2: convert to segmentation
        segmentation = self.label_manager.convert_logits_to_segmentation(avg_probs)  # shape: (H, W, D)

        # Step 3: wrap as MetaTensor and attach meta
        seg_tensor = MetaTensor(segmentation[None].astype(np.uint8))  # add channel dim
        seg_tensor.meta = dict(meta)

        # Replace the key or store in new key
        d[self.output_key] = seg_tensor
        print('Data keys output: ', d.keys())
        return d

class ModelnnUNetWrapper(torch.nn.Module):
    """
    A wrapper class for nnUNet model integration with MONAI framework.
    The wrapper can be use to integrate the nnUNet Bundle within MONAI framework for inference.

    Parameters
    ----------
    predictor : nnUNetPredictor
        The nnUNet predictor object used for inference.
    model_folder : Union[str, Path]
        The folder path where the model and related files are stored.
    model_name : str, optional
        The name of the model file, by default "model.pt".
    dataset_json : dict, optional
        The dataset JSON file containing dataset information.
    plans : dict, optional
        The plans JSON file containing model configuration.
    nnunet_config : dict, optional
        The nnUNet configuration dictionary containing model parameters.

    Attributes
    ----------
    predictor : nnUNetPredictor
        The nnUNet predictor object used for inference.
    network_weights : torch.nn.Module
        The network weights of the model.

    Notes
    -----
    This class integrates nnUNet model with MONAI framework by loading necessary configurations,
    restoring network architecture, and setting up the predictor for inference.
    """

    def __init__(self, predictor: object,
                 model_folder: Union[str, Path],
                 checkpoint_name: str = None,
                 dataset_json: dict = None, 
                 plans: dict = None,
                 nnunet_config: dict = None,
                 save_probabilities: bool = False,
                 save_files: bool = False,
                 tmp_dir: str = 'tmp',
                 use_folds: Union[int, str, Tuple[Union[int, str], ...], List[Union[int, str]]] = None):
                 
        super().__init__()
        self.predictor = predictor

        model_training_output_dir = model_folder
        model_parent_dir = Path(model_training_output_dir).parent
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        
        print('wrapper: Model Folder: ', model_training_output_dir)
        print('wrapper: Model name: ', checkpoint_name)
        print('wrapper: Model parent dir: ', model_parent_dir)
        print('wrapper:use_folds: ',use_folds)
        if not checkpoint_name:
            raise ValueError("Model name is required. Please provide a valid model name.")
        
 # Block Added from nnUNet/nnunetv2/inference/predict_from_raw_data.py#nnUNetPredictor
        if dataset_json is None:
            dataset_json = load_json(join(Path(model_parent_dir), "dataset.json"))
        if plans is None:
            plans = load_json(join(Path(model_parent_dir), "plans.json"))
        plans_manager = PlansManager(plans)

        parameters = []
        
        # self.model_subpath = os.path.dirname(model_name)
        # self.fold_name = ''
        # if self.model_subpath:
        #     parts = self.model_subpath.split('/')
        #     if len(parts) > 1:
        #         self.fold_name = parts[-1] 
        
        self.tmp_dir = tmp_dir
        
        if nnunet_config is None:
            checkpoint_path = join(Path(model_training_output_dir), "nnunet_checkpoint.pth")
            if not os.path.exists(checkpoint_path):
                # If the checkpoint does not exist, raise an error
                raise ValueError(f"Checkpoint file not found at {checkpoint_path}. Please ensure the model is trained and the checkpoint exists.")
            checkpoint_path = join(Path(model_training_output_dir), "nnunet_checkpoint.pth")
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
            trainer_name = checkpoint["trainer_name"]
            configuration_name = checkpoint["init_args"]["configuration"]
            inference_allowed_mirroring_axes = (
                checkpoint["inference_allowed_mirroring_axes"]
                if "inference_allowed_mirroring_axes" in checkpoint.keys()
                else None
            )
        else:
            trainer_name = nnunet_config["trainer_name"]
            configuration_name = nnunet_config["configuration"]
            inference_allowed_mirroring_axes = nnunet_config["inference_allowed_mirroring_axes"]
        
        
        # Auto detect folds
        if isinstance(use_folds, str) or isinstance(use_folds, int):
            use_folds = [use_folds]
        
        if use_folds is None:
            use_folds = self.predictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
        
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            monai_checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if "network_weights" in monai_checkpoint.keys():
                parameters.append(monai_checkpoint["network_weights"])
            else:
                parameters.append(monai_checkpoint)
            
        configuration_manager = plans_manager.get_configuration(configuration_name)
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer_name, "nnunetv2.training.nnUNetTrainer"
        )
        if trainer_class is None:
            raise RuntimeError(
                f"Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. "
                f"Please place it there (in any .py file)!"
            )
        
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )

        predictor.plans_manager = plans_manager  # type: ignore
        predictor.configuration_manager = configuration_manager  # type: ignore
        predictor.list_of_parameters = parameters  # type: ignore
        
        print('Len of parameters/folds: ', len(parameters))
        print('Configuration name: ', configuration_name)

        #network.load_state_dict(parameters[0])
        predictor.network = network  # type: ignore
        predictor.dataset_json = dataset_json  # type: ignore
        predictor.trainer_name = trainer_name  # type: ignore
        predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes  # type: ignore
        predictor.label_manager = plans_manager.get_label_manager(dataset_json)  # type: ignore

        self.network_weights = self.predictor.network  # type: ignore
        
        self.save_probabilities = save_probabilities
        self.save_files = save_files
        self.configuration_name = configuration_name

    def forward(self, x: MetaTensor) -> MetaTensor:
        """
        Forward pass for the nnUNet model.

        Args:
            x (MetaTensor): Input tensor. If the input is a tuple,
                it is assumed to be a decollated batch (list of tensors). Otherwise, it is assumed to be a collated batch.

        Returns:
            MetaTensor: The output tensor with the same metadata as the input.

        Raises:
            TypeError: If the input is not a torch.Tensor or a tuple of MetaTensors.

        Notes:
            - If the input is a tuple, the filenames are extracted from the metadata of each tensor in the tuple.
            - If the input is a collated batch, the filenames are extracted from the metadata of the input tensor.
            - The filenames are used to generate predictions using the nnUNet predictor.
            - The predictions are converted to torch tensors, with added batch and channel dimensions.
            - The output tensor is concatenated along the batch dimension and returned as a MetaTensor with the same metadata.
        """
        if isinstance(x, MetaTensor):
            # # print x shape
            # print(f'bundle: Input shape (x): {x.shape}')
            # # print x.meta['affine'] shape
            print(f'bundle: Input affine shape (x.meta["affine"]): {x.meta["affine"].shape}')
            print(f'bundle: Input affine (x.meta["affine"]): {x.meta["affine"]}')
            print(f'bundle: Input affine (x.meta["affine"][0] shape): {x.meta["affine"][0].shape}')

            # print(f'bundle: Input affine (x.meta["pixdim"]): {x.meta["pixdim"]}')

            spatial_shape = list(x.shape[-3:])  # [H, W, D] or [X, Y, Z]
            if "pixdim" in x.meta:
                print('Getting spacing from pixdim...')
                if x.meta["pixdim"].ndim == 1:
                    properties_or_list_of_properties = {"spacing": x.meta["pixdim"][1:4].tolist()}
                else:
                    properties_or_list_of_properties = {"spacing": x.meta["pixdim"][0][1:4].numpy().tolist()}
            elif "affine" in x.meta:
                print('Getting spacing from affine matrix...')
                spacing = [
                    abs(x.meta["affine"][0][0][0].item()),
                    abs(x.meta["affine"][0][1][1].item()),
                    abs(x.meta["affine"][0][2][2].item()),
                ]
                properties_or_list_of_properties = {"spacing": spacing}
            else:
                properties_or_list_of_properties = {"spacing": [1.0, 1.0, 1.0]}
            properties_or_list_of_properties["spatial_shape"] = spatial_shape
        else:
            raise TypeError("Input must be a MetaTensor or a tuple of MetaTensors.")

        # print properties_or_list_of_properties
        print(f'Input properties: {properties_or_list_of_properties}')

        # Print x type and x shape
        print(f'Input type (x): {type(x)}, shape: {x.shape}')

        image_or_list_of_images = x.cpu().numpy()[0, :]

        # Print shape of image
        # if list, print shape of each image
        if isinstance(image_or_list_of_images, list):
            # Print first image shape
            print(f'Shape of preprocessed image 0: ', image_or_list_of_images[0].shape)
        else:
            print('Shape of preprocessed image: ', image_or_list_of_images.shape)

        ## Add error catching later - Elan
        if self.save_files:
            # Save the input image to a file
            infile = x.meta["filename_or_obj"]
            if isinstance(infile, list):
                infile = infile[0]
            outfile_name = os.path.basename(infile).split('.')[0]
            outfolder =  Path(self.tmp_dir).joinpath(self.configuration_name)
            os.makedirs(outfolder, exist_ok=True)
            outfile = str(Path(outfolder).joinpath(outfile_name))
            print(' Saving files to: ', outfile)

            # Extract 4x4 affine matrix from metadata if affine shape is [1,4,4]
            # Need to make sure this works for 2D images
            affine = None
            if x.meta['affine'].shape == (1, 4, 4):
                affine = x.meta['affine'][0].cpu().numpy()  # shape (4, 4)
            elif x.meta['affine'].shape == (4, 4):
                affine = x.meta['affine'].cpu().numpy()  # shape (4, 4)
            else:
                raise ValueError(f"Unexpected affine shape: {x.meta['affine'].shape}")

            # Extract spacing as norm of each column in affine
            spacing = tuple(np.linalg.norm(affine[:3, i]) for i in range(3))

            # Extract origin (translation component)
            origin = tuple(float(v) for v in affine[:3, 3])

            # Normalize the direction matrix (remove spacing influence)
            direction_matrix = affine[:3, :3] / spacing
            direction = tuple(direction_matrix.flatten().round(6))  # ensure clean rounding

            # Assign to output dict
            properties_or_list_of_properties['sitk_stuff'] = {
                'spacing': spacing,
                'origin': origin,
                'direction': direction
            }
        else:
            outfile = None
        
         # if 3d_cascade_fullres, load segmentation from 3d_lowres
        previous_segmentation = None
        if self.configuration_name == "3d_cascade_fullres":
            # load segmentation from 3d_lowres
            lowres_predictions_folder = os.path.join(self.tmp_dir,"3d_lowres")

            if outfile:
                seg_file = os.path.join(lowres_predictions_folder, outfile_name + '.nii.gz')
                # load the lowres segmentation from nifti
                rw = self.predictor.plans_manager.image_reader_writer_class()
                previous_segmentation, _ = rw.read_seg(seg_file)

                if previous_segmentation is None:
                    raise ValueError("Failed to load previous segmentation.")
            else:
                raise ValueError("Output file name is required for 3d_cascade_fullres configuration.")
                
        
        # input_files should be a list of file paths, one per modality
        prediction_output = self.predictor.predict_from_list_of_npy_arrays(  # type: ignore
            image_or_list_of_images,
            previous_segmentation,
            properties_or_list_of_properties,
            save_probabilities=self.save_probabilities,
            truncated_ofname=outfile,
            num_processes=2,
            num_processes_segmentation_export=2,
        )
        
        if not self.save_files:
            out_tensors = []
            for out in prediction_output:  # Add batch and channel dimensions
                out_tensors.append(torch.from_numpy(np.expand_dims(np.expand_dims(out, 0), 0)))
            out_tensor = torch.cat(out_tensors, 0)  # Concatenate along batch dimension
            
            return MetaTensor(out_tensor, meta=x.meta)
        else:
            saved_path =  outfile + '.npz'
            # Check if the file exists
            if not os.path.exists(saved_path):
                raise FileNotFoundError(f"File not found: {saved_path}")
            
            # Load the properties from the saved file
            shape = properties_or_list_of_properties['spatial_shape']
            
            # Set channels and batch size
            channels = 1
            batch_size = 1

            # Create a zero tensor
            dummy_tensor = torch.zeros((batch_size, channels, *shape), dtype=torch.float32)
            # Create a new meta dict with the file path added
            meta_with_filepath = dict(x.meta)
            meta_with_filepath["saved_file"] = saved_path
            
            print('Returning dummy tensor with saved file path: ', saved_path)
            
            return MetaTensor(dummy_tensor, meta=meta_with_filepath)