#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import torchf
import os
from utils.common import (
    m2f_dataset_collate,
    m2f_extract_pred_maps_and_masks,
    get_perpixel_features,
    set_seed,
    pixel_mean_std,
    CADIS_PIXEL_MEAN,
    CADIS_PIXEL_STD,
    CAT1K_PIXEL_MEAN,
    CAT1K_PIXEL_STD,
    WEIGHTS_CADIS_TRAIN
)
from utils.dataset_utils import (
    get_cadisv2_dataset,
    get_cataract1k_dataset,
    ZEISS_CATEGORIES,
)
from utils.augmentations import train_transforms_color_jitter
from utils.medical_datasets import Mask2FormerDataset
from transformers import (
    Mask2FormerForUniversalSegmentation,
    SwinModel,
    SwinConfig,
    Mask2FormerConfig,
    AutoImageProcessor,
    Mask2FormerImageProcessor,
)
from torch.utils.data import DataLoader
import evaluate
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dotenv import load_dotenv
import wandb
from copy import deepcopy
import shutil
import random
from utils.wandb_utils import log_table_of_images
from utils.losses import PixelContrastLoss


# In[3]:


NUM_CLASSES = len(ZEISS_CATEGORIES) - 3  + 1 # Remove class incremental and add background !!!
SWIN_BACKBONE = "microsoft/swin-tiny-patch4-window7-224"#"microsoft/swin-large-patch4-window12-384"

# Download pretrained swin model
swin_model = SwinModel.from_pretrained(
    SWIN_BACKBONE, out_features=["stage1", "stage2", "stage3", "stage4"]
)
swin_config = SwinConfig.from_pretrained(
    SWIN_BACKBONE, out_features=["stage1", "stage2", "stage3", "stage4"]
)

# Create Mask2Former configuration based on Swin's configuration
mask2former_config = Mask2FormerConfig(
    backbone_config=swin_config, num_labels=NUM_CLASSES #, ignore_value=BG_VALUE
)

# Create the Mask2Former model with this configuration
model = Mask2FormerForUniversalSegmentation(mask2former_config)

# Reuse pretrained parameters
for swin_param, m2f_param in zip(
    swin_model.named_parameters(),
    model.model.pixel_level_module.encoder.named_parameters(),
):
    m2f_param_name = f"model.pixel_level_module.encoder.{m2f_param[0]}"

    if swin_param[0] == m2f_param[0]:
        model.state_dict()[m2f_param_name].copy_(swin_param[1])
        continue

    print(f"Not Matched: {m2f_param[0]} != {swin_param[0]}")


# In[4]:


# Helper function to load datasets
def load_dataset(dataset_getter, data_path, domain_incremental):
    return dataset_getter(data_path, domain_incremental=domain_incremental)


# Helper function to create dataloaders for a dataset
def create_dataloaders(
    dataset, batch_size, shuffle, num_workers, drop_last, pin_memory, collate_fn
):
    batch_size_val=16
    batch_size_test=16

    return {
        "train": DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            dataset["val"],
            batch_size=batch_size_val,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            dataset["test"],
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
    }


# Load datasets
datasets = {
    "A": load_dataset(get_cadisv2_dataset, "../../storage/data/CaDISv2", True),
    "B": load_dataset(get_cataract1k_dataset, "../../storage/data/cataract-1k", True),
}


pixel_mean_A=np.array(CADIS_PIXEL_MEAN)
pixel_std_A=np.array(CADIS_PIXEL_STD)
pixel_mean_B=np.array(CAT1K_PIXEL_MEAN)
pixel_std_B=np.array(CAT1K_PIXEL_STD)

weights_A=WEIGHTS_CADIS_TRAIN # Class weights used for weightes contrastive loss

# Define preprocessor
swin_processor = AutoImageProcessor.from_pretrained(SWIN_BACKBONE)
m2f_preprocessor_A = Mask2FormerImageProcessor(
    reduce_labels=False,
    ignore_index=255,
    do_resize=False,
    do_rescale=False,
    do_normalize=True,
    image_std=pixel_std_A,
    image_mean=pixel_mean_A,
)

m2f_preprocessor_B = Mask2FormerImageProcessor(
    reduce_labels=False,
    ignore_index=255,
    do_resize=False,
    do_rescale=False,
    do_normalize=True,
    image_std=pixel_std_B,
    image_mean=pixel_mean_B,
)
# Create Mask2Former Datasets

m2f_datasets = {
    "A": {
        "train": Mask2FormerDataset(datasets["A"][0], m2f_preprocessor_A, transform=train_transforms_color_jitter),
        "val": Mask2FormerDataset(datasets["A"][1], m2f_preprocessor_A),
        "test": Mask2FormerDataset(datasets["A"][2], m2f_preprocessor_A),
    },
    "B": {
        "train": Mask2FormerDataset(datasets["B"][0], m2f_preprocessor_B, transform=train_transforms_color_jitter),
        "val": Mask2FormerDataset(datasets["B"][1], m2f_preprocessor_B),
        "test": Mask2FormerDataset(datasets["B"][2], m2f_preprocessor_B),
    },
}

# DataLoader parameters
N_WORKERS = 4
BATCH_SIZE = 16
SHUFFLE = True
DROP_LAST = True

dataloader_params = {
    "batch_size": BATCH_SIZE,
    "shuffle": SHUFFLE,
    "num_workers": N_WORKERS,
    "drop_last": DROP_LAST,
    "pin_memory": True,
    "collate_fn": m2f_dataset_collate,
}

# Create DataLoaders
dataloaders = {
    key: create_dataloaders(m2f_datasets[key], **dataloader_params)
    for key in m2f_datasets
}

print(dataloaders)


# In[5]:


# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

BG_VALUE_255=255
base_run_name="M2F-Swin-Tiny-Train_Cadis_Contrastive_Loss_Aug"
project_name = "M2F_latest"
user_or_team = "continual-learning-tum"
new_run_name="M2F-Swin-Tiny-Train_Naive-Forgetting-CL-Aug"


# In[6]:


# Tensorboard setup
out_dir="outputs/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_dir+"runs"):
    os.makedirs(out_dir+"runs")
# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir outputs/runs')


# In[7]:


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# In[33]:


# Tensorboard logging
writer = SummaryWriter(log_dir=out_dir + "runs")

# Model checkpointing
model_dir = out_dir + "models/"
if not os.path.exists(model_dir):
    print("Store weights in: ", model_dir)
    os.makedirs(model_dir)

best_model_dir = model_dir + f"{base_run_name}/best_model/"
if not os.path.exists(best_model_dir):
    print("Store best model weights in: ", best_model_dir)
    os.makedirs(best_model_dir)
final_model_dir = model_dir + f"{base_run_name}/final_model/"
if not os.path.exists(final_model_dir):
    print("Store final model weights in: ", final_model_dir)
    os.makedirs(final_model_dir)


# In[9]:


# WandB for team usage !!!!

wandb.login() # use this one if a different person is going to run the notebook
#wandb.login(relogin=False) # if the same person in the last run is going to run the notebook again


# In[37]:


# Training
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
LR_MULTIPLIER = 0.1
BACKBONE_LR = LEARNING_RATE * LR_MULTIPLIER
WEIGHT_DECAY = 0.05
PATIENCE=15

# Important to ALWAYS set the full_batch_sampling=True for A training (no replay)

use_prototypes=False # Change it to True if prototypes will be used as anchors and change the loss function as below: 
#contrastive_loss=PixelContrastLoss(full_batch_sampling=True, num_classes=NUM_CLASSES, num_prototypes_per_class=5, in_channels=256)

#contrastive_loss = PixelContrastLoss(full_batch_sampling=True, weights=weights_A) # if weighted contrastive loss needs to be used
contrastive_loss=PixelContrastLoss(full_batch_sampling=True) # vanilla CL
CONTRASTIVE_LOSS_LAMBDA=1 # Increase the lambda when necessary (probably while using weighted contrastive loss)


# In[11]:


metric = evaluate.load("mean_iou")
encoder_params = [
    param
    for name, param in model.named_parameters()
    if name.startswith("model.pixel_level_module.encoder")
]
decoder_params = [
    param
    for name, param in model.named_parameters()
    if name.startswith("model.pixel_level_module.decoder")
]
transformer_params = [
    param
    for name, param in model.named_parameters()
    if name.startswith("model.transformer_module")
]
class_prediction_params=[
    param
    for name, param in model.named_parameters() 
    if not name.startswith("model.pixel_level_module.encoder") and not name.startswith("model.transformer_module") and not name.startswith("model.pixel_level_module.decoder")
]
optimizer = optim.AdamW(
    [
        {"params": encoder_params, "lr": BACKBONE_LR},
        {"params": decoder_params},
        {"params": transformer_params},
        {"params": class_prediction_params}
    ],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=NUM_EPOCHS, power=0.9
)


# In[ ]:


wandb.init(
    project=project_name,
    config={
        "learning_rate": LEARNING_RATE,
        "learning_rate_multiplier": LR_MULTIPLIER,
        "backbone_learning_rate": BACKBONE_LR,
        "learning_rate_scheduler": scheduler.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "backbone": SWIN_BACKBONE,
        "m2f_preprocessor": m2f_preprocessor_A.__dict__,
        "m2f_model_config": model.config
    },
    name=new_run_name,
    notes="M2F with tiny Swin backbone pretrained on ImageNet-1K. \
        Scenario: Train on A, Test on A"
)
print("wandb run id:",wandb.run.id)


# ## Test on A

# In[12]:


# Load best model and evaluate on test
# Construct the artifact path
artifact_path = f"{user_or_team}/{project_name}/best_model_{base_run_name}:latest"

# Load from W&B
api = wandb.Api()
artifact=api.artifact(artifact_path)
artifact_dir=artifact.download()
model_state_dict_path = os.path.join(artifact_dir, f"best_model_{base_run_name}.pth" )
model_state_dict = torch.load(model_state_dict_path,map_location=device)
model = Mask2FormerForUniversalSegmentation(mask2former_config)
model.load_state_dict(model_state_dict)
model.to(device)


# In[ ]:


model.eval()
test_running_loss = 0
CURR_TASK="A"
test_loader = tqdm(dataloaders[CURR_TASK]["test"], desc="Test loop")

BATCH_INDEX = 0
table = wandb.Table(columns=["ID", "Image"])
with torch.no_grad():
    for batch in test_loader:
        # Move everything to the device
        batch["pixel_values"] = batch["pixel_values"].to(device)
        batch["pixel_mask"] = batch["pixel_mask"].to(device)
        batch["mask_labels"] = [entry.to(device) for entry in batch["mask_labels"]]
        batch["class_labels"] = [entry.to(device) for entry in batch["class_labels"]]
        # Compute output and loss
        outputs = model(**batch)

        loss = outputs.loss
        # Record losses
        current_loss = loss.item() * batch["pixel_values"].size(0)
        test_running_loss += current_loss
        test_loader.set_postfix(loss=f"{current_loss:.4f}")

        # Extract and compute metrics
        pred_maps, masks = m2f_extract_pred_maps_and_masks(
            batch, outputs, m2f_preprocessor_A
        )
        metric.add_batch(references=masks, predictions=pred_maps)
        if BATCH_INDEX <5:
            # Visualize
            log_table_of_images(
                table, # common table for all batches
                batch["pixel_values"],
                pixel_mean_A, # remove normalization
                pixel_std_A, # remove normalization
                pred_maps,
                masks,
                BATCH_INDEX, # correct indexing in table
            )
            BATCH_INDEX += 1
# Log table
wandb.log({f"{CURR_TASK}_TEST_AFTER_TRAINING_A": table})

# After compute the batches that were added are deleted
test_metrics_A = metric.compute(
    num_labels=NUM_CLASSES, ignore_index=BG_VALUE_255, reduce_labels=False
)
mean_test_iou = test_metrics_A["mean_iou"]
final_test_loss = test_running_loss / len(dataloaders[CURR_TASK]["test"].dataset)
wandb.log({
    f"Loss/test_{CURR_TASK}": final_test_loss,
    f"mIoU/test_{CURR_TASK}": mean_test_iou
})


# ## Test on B

# In[ ]:


model.eval()
test_running_loss = 0
CURR_TASK="B"
test_loader = tqdm(dataloaders[CURR_TASK]["test"], desc="Test loop")

BATCH_INDEX = 0
table = wandb.Table(columns=["ID", "Image"])
with torch.no_grad():
    for batch in test_loader:
        # Move everything to the device
        batch["pixel_values"] = batch["pixel_values"].to(device)
        batch["pixel_mask"] = batch["pixel_mask"].to(device)
        batch["mask_labels"] = [entry.to(device) for entry in batch["mask_labels"]]
        batch["class_labels"] = [entry.to(device) for entry in batch["class_labels"]]
        # Compute output and loss
        outputs = model(**batch)

        loss = outputs.loss
        # Record losses
        current_loss = loss.item() * batch["pixel_values"].size(0)
        test_running_loss += current_loss
        test_loader.set_postfix(loss=f"{current_loss:.4f}")

        # Extract and compute metrics
        pred_maps, masks = m2f_extract_pred_maps_and_masks(
            batch, outputs, m2f_preprocessor_B
        )
        metric.add_batch(references=masks, predictions=pred_maps)
        if BATCH_INDEX <5:
            # Visualize
            log_table_of_images(
                table, # common table for all batches
                batch["pixel_values"],
                np.array(CAT1K_PIXEL_MEAN), # remove normalization
                np.array(CAT1K_PIXEL_STD), # remove normalization
                pred_maps,
                masks,
                BATCH_INDEX, # correct indexing in table
            )
            BATCH_INDEX += 1

# Log table
wandb.log({f"{CURR_TASK}_TEST_AFTER_TRAINING_A": table})

# After compute the batches that were added are deleted
test_metrics_B_before = metric.compute(
    num_labels=NUM_CLASSES, ignore_index=BG_VALUE_255, reduce_labels=False
)
mean_test_iou = test_metrics_B_before["mean_iou"]
final_test_loss = test_running_loss / len(dataloaders[CURR_TASK]["test"].dataset)
wandb.log({
    f"Loss/test_{CURR_TASK}": final_test_loss,
    f"mIoU/test_{CURR_TASK}": mean_test_iou
})
print(f"Test Loss: {final_test_loss:.4f}, Test mIoU: {mean_test_iou:.4f}")


# # Train on B now with CL only

# In[13]:


metric = evaluate.load("mean_iou")
encoder_params = [
    param
    for name, param in model.named_parameters()
    if name.startswith("model.pixel_level_module.encoder")
]
decoder_params = [
    param
    for name, param in model.named_parameters()
    if name.startswith("model.pixel_level_module.decoder")
]
transformer_params = [
    param
    for name, param in model.named_parameters()
    if name.startswith("model.transformer_module")
]
class_prediction_params=[
    param
    for name, param in model.named_parameters() 
    if not name.startswith("model.pixel_level_module.encoder") and not name.startswith("model.transformer_module") and not name.startswith("model.pixel_level_module.decoder")
]
optimizer = optim.AdamW(
    [
        {"params": encoder_params, "lr": BACKBONE_LR},
        {"params": decoder_params},
        {"params": transformer_params},
        {"params": class_prediction_params}
    ],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=NUM_EPOCHS, power=0.9
)


# In[ ]:


# To avoid making stupid errors
CURR_TASK = "B"
if use_prototypes:
    os.makedirs(f"{best_model_dir}{CURR_TASK}_prototypes/",exist_ok=True)

# For storing the model
best_val_metric = -np.inf
best_model_weights=None # best model weights are stored here
best_prototypes=None

# Move model to device
model.to(device)
counter=0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_running_loss = 0.0
    val_running_loss = 0.0

    # Set up tqdm for the training loop
    train_loader = tqdm(
        dataloaders[CURR_TASK]["train"], desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Training"
    )

    for batch in train_loader:
        # Move everything to the device
        batch["pixel_values"] = batch["pixel_values"].to(device)
        batch["pixel_mask"] = batch["pixel_mask"].to(device)
        batch["mask_labels"] = [entry.to(device) for entry in batch["mask_labels"]]
        batch["class_labels"] = [entry.to(device) for entry in batch["class_labels"]]
       
        outputs = model(**batch,output_hidden_states=True)
        # Extract and compute metrics
        pred_maps, masks = m2f_extract_pred_maps_and_masks(
            batch, outputs, m2f_preprocessor_A
        )
        metric.add_batch(references=masks, predictions=pred_maps)

        feats=get_perpixel_features(outputs.pixel_decoder_hidden_states,outputs.pixel_decoder_last_hidden_state).to(device)
        contrastive_loss_output=contrastive_loss(feats,labels=torch.stack(masks,dim=0).to(device),predict=torch.stack(pred_maps,dim=0).to(device))
        
        loss = outputs.loss + CONTRASTIVE_LOSS_LAMBDA * contrastive_loss_output
        
        # Compute gradient and perform step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record losses
        current_loss = loss.item() * batch["pixel_values"].size(0)
        train_running_loss += current_loss
        train_loader.set_postfix(loss=f"{current_loss:.4f}")
        

        
    
    # After compute the batches that were added are deleted
    mean_train_iou = metric.compute(
        num_labels=NUM_CLASSES, ignore_index=BG_VALUE_255, reduce_labels=False
    )["mean_iou"]

    # Validation phase
    model.eval()
    val_loader = tqdm(
        dataloaders[CURR_TASK]["val"], desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Validation"
    )
    with torch.no_grad():
        for batch in val_loader:
            # Move everything to the device
            batch["pixel_values"] = batch["pixel_values"].to(device)
            batch["pixel_mask"] = batch["pixel_mask"].to(device)
            batch["mask_labels"] = [entry.to(device) for entry in batch["mask_labels"]]
            batch["class_labels"] = [
                entry.to(device) for entry in batch["class_labels"]
            ]
            # Compute output and loss
            outputs = model(**batch)

            loss = outputs.loss
            # Record losses
            current_loss = loss.item() * batch["pixel_values"].size(0)
            val_running_loss += current_loss
            val_loader.set_postfix(loss=f"{current_loss:.4f}")

            # Extract and compute metrics
            pred_maps, masks = m2f_extract_pred_maps_and_masks(
                batch, outputs, m2f_preprocessor_A
            )
            metric.add_batch(references=masks, predictions=pred_maps)
            

    # After compute the batches that were added are deleted
    mean_val_iou = metric.compute(
        num_labels=NUM_CLASSES, ignore_index=BG_VALUE_255, reduce_labels=False
    )["mean_iou"]

    epoch_train_loss = train_running_loss / len(dataloaders[CURR_TASK]["train"].dataset)
    epoch_val_loss = val_running_loss / len(dataloaders[CURR_TASK]["val"].dataset)

    writer.add_scalar(f"Loss/train_{new_run_name}_{CURR_TASK}", epoch_train_loss, epoch + 1)
    writer.add_scalar(f"Loss/val_{new_run_name}_{CURR_TASK}", epoch_val_loss, epoch + 1)
    writer.add_scalar(f"mIoU/train_{new_run_name}_{CURR_TASK}", mean_train_iou, epoch + 1)
    writer.add_scalar(f"mIoU/val_{new_run_name}_{CURR_TASK}", mean_val_iou, epoch + 1)

    wandb.log({
        f"Loss/train_replay_A_{CURR_TASK}": epoch_train_loss,
        f"Loss/val_replay_A_{CURR_TASK}": epoch_val_loss,
        f"mIoU/train_replay_A_{CURR_TASK}": mean_train_iou,
        f"mIoU/val_replay_A_{CURR_TASK}": mean_val_iou
    })


    tqdm.write(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.4f}, Train mIoU: {mean_train_iou:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation mIoU: {mean_val_iou:.4f}"
    )
    
    if mean_val_iou > best_val_metric:
        best_val_metric = mean_val_iou
        model.save_pretrained(f"{best_model_dir}{CURR_TASK}/")
        best_model_weights = deepcopy(model.state_dict())
        counter=0
        if use_prototypes:
            best_prototypes=deepcopy(contrastive_loss.prototypes)
            torch.save(best_prototypes, f"{best_model_dir}{CURR_TASK}_prototypes/prototypes.pth")
        
    else:
        counter+=1
        if counter == PATIENCE:
            print("Early stopping at epoch",epoch)
            break
            
os.makedirs(f"{best_model_dir}{CURR_TASK}/",exist_ok=True)
artifact = wandb.Artifact(f"best_model_{new_run_name}", type="model")
trained_model_path=f"{best_model_dir}{CURR_TASK}/best_model_{new_run_name}.pth"
artifact.add_file(trained_model_path, torch.save(best_model_weights, trained_model_path))

if use_prototypes:
    prototypes_path = f"{best_model_dir}{CURR_TASK}/prototypes_{new_run_name}.pth"
    artifact.add_file(prototypes_path,torch.save(best_prototypes,prototypes_path))
    
wandb.run.log_artifact(artifact)

if os.path.exists(trained_model_path):
    os.remove(trained_model_path)

if use_prototypes:
    if os.path.exists(prototypes_path):
        os.remove(prototypes_path)



# In[ ]:


# model = Mask2FormerForUniversalSegmentation.from_pretrained(f"{best_model_dir}{CURR_TASK}/").to(device)

# Load best model and evaluate on test
# Construct the artifact path
artifact_path = f"{user_or_team}/{project_name}/best_model_{new_run_name}:latest"

# Load from W&B
api = wandb.Api()
artifact=api.artifact(artifact_path)
artifact_dir=artifact.download()
model_state_dict_path = os.path.join(artifact_dir, f"best_model_{new_run_name}.pth" )
model_state_dict = torch.load(model_state_dict_path,map_location=device)
model = Mask2FormerForUniversalSegmentation(mask2former_config)
model.load_state_dict(model_state_dict)
model.to(device)


# # Test on B

# In[ ]:


model.eval()
test_running_loss = 0
CURR_TASK="B"
test_loader = tqdm(dataloaders[CURR_TASK]["test"], desc="Test loop")
BATCH_INDEX = 0
table = wandb.Table(columns=["ID", "Image"])
with torch.no_grad():
    for batch in test_loader:
        # Move everything to the device
        batch["pixel_values"] = batch["pixel_values"].to(device)
        batch["pixel_mask"] = batch["pixel_mask"].to(device)
        batch["mask_labels"] = [entry.to(device) for entry in batch["mask_labels"]]
        batch["class_labels"] = [entry.to(device) for entry in batch["class_labels"]]
        # Compute output and loss
        outputs = model(**batch)

        loss = outputs.loss
        # Record losses
        current_loss = loss.item() * batch["pixel_values"].size(0)
        test_running_loss += current_loss
        test_loader.set_postfix(loss=f"{current_loss:.4f}")

        # Extract and compute metrics
        pred_maps, masks = m2f_extract_pred_maps_and_masks(
            batch, outputs, m2f_preprocessor_B
        )
        metric.add_batch(references=masks, predictions=pred_maps)
        if BATCH_INDEX <5:
            # Visualize
            log_table_of_images(
                table, # common table for all batches
                batch["pixel_values"],
                pixel_mean_B, # remove normalization
                pixel_std_B, # remove normalization
                pred_maps,
                masks,
                BATCH_INDEX, # correct indexing in table
            )
            BATCH_INDEX += 1
# Log table
wandb.log({f"{CURR_TASK}_TEST_AFTER_TRAINING_B": table})

# After compute the batches that were added are deleted
test_metrics_B = metric.compute(
    num_labels=NUM_CLASSES, ignore_index=BG_VALUE_255, reduce_labels=False
)
mean_test_iou = test_metrics_B["mean_iou"]
final_test_loss = test_running_loss / len(dataloaders[CURR_TASK]["test"].dataset)
wandb.log({
    f"Loss/test_{CURR_TASK}": final_test_loss,
    f"mIoU/test_{CURR_TASK}": mean_test_iou
})
print(f"Test Loss: {final_test_loss:.4f}, Test mIoU: {mean_test_iou:.4f}")


# # Test on A

# In[ ]:


# To avoid making stupid errors
CURR_TASK = "A"

model.eval()
test_running_loss = 0
test_loader = tqdm(dataloaders[CURR_TASK]["test"], desc="Test loop")
BATCH_INDEX = 0
table = wandb.Table(columns=["ID", "Image"])
with torch.no_grad():
    for batch in test_loader:
        # Move everything to the device
        batch["pixel_values"] = batch["pixel_values"].to(device)
        batch["pixel_mask"] = batch["pixel_mask"].to(device)
        batch["mask_labels"] = [entry.to(device) for entry in batch["mask_labels"]]
        batch["class_labels"] = [entry.to(device) for entry in batch["class_labels"]]
        # Compute output and loss
        outputs = model(**batch)

        loss = outputs.loss
        # Record losses
        current_loss = loss.item() * batch["pixel_values"].size(0)
        test_running_loss += current_loss
        test_loader.set_postfix(loss=f"{current_loss:.4f}")

        # Extract and compute metrics
        pred_maps, masks = m2f_extract_pred_maps_and_masks(
            batch, outputs, m2f_preprocessor_A
        )
        metric.add_batch(references=masks, predictions=pred_maps)
        if BATCH_INDEX <5:
            # Visualize
            log_table_of_images(
                table, # common table for all batches
                batch["pixel_values"],
                pixel_mean_A, # remove normalization
                pixel_std_A, # remove normalization
                pred_maps,
                masks,
                BATCH_INDEX, # correct indexing in table
            )
            BATCH_INDEX += 1
# Log table
wandb.log({f"{CURR_TASK}_TEST_AFTER_TRAINING_B": table})

# After compute the batches that were added are deleted
test_metrics_forgetting_A = metric.compute(
    num_labels=NUM_CLASSES, ignore_index=BG_VALUE_255, reduce_labels=False
)
mean_test_iou = test_metrics_forgetting_A["mean_iou"]
final_test_loss = test_running_loss / len(dataloaders[CURR_TASK]["test"].dataset)
wandb.log({
    f"Loss/test_naive_forgetting_{CURR_TASK}": final_test_loss,
    f"mIoU/test_naive_forgetting_{CURR_TASK}": mean_test_iou
})
print(f"Test Loss: {final_test_loss:.4f}, Test mIoU: {mean_test_iou:.4f}")


# # Evaluate

# In[ ]:


# Collect overall mIoU
mIoU_A_before = test_metrics_A["mean_iou"]
mIoU_B_before=test_metrics_B_before["mean_iou"]
mIoU_forgetting_A = test_metrics_forgetting_A["mean_iou"]
mIoU_B = test_metrics_B["mean_iou"]

# Collect per category mIoU
per_category_mIoU_A_before = np.array(test_metrics_A["per_category_iou"])
per_category_mIoU_A = np.array(test_metrics_forgetting_A["per_category_iou"])
per_category_mIoU_B = np.array(test_metrics_B["per_category_iou"])
per_category_mIoU_B_before=np.array(test_metrics_B_before["per_category_iou"])

# Average learning accuracies (mIoUs)
avg_learning_acc = (mIoU_A_before + mIoU_B) / 2
per_category_avg_learning_acc = (per_category_mIoU_A_before + per_category_mIoU_B) / 2

# Forgetting
total_forgetting = mIoU_A_before - mIoU_forgetting_A
per_category_forgetting = (per_category_mIoU_A_before - per_category_mIoU_A)

# Export evaluation metrics to WandB
wandb.log({
    "eval/avg_learning_acc": avg_learning_acc,
    "eval/total_forgetting": total_forgetting,
})

columns=["categories","per_category_mIoU_A_before","per_category_mIoU_B_before",
         "per_category_mIoU_B", "per_category_mIoU_A",
         "per_category_avg_learning_acc","per_category_forgetting"]
data=[]

data.append(["background",per_category_mIoU_A_before[0],
                 per_category_mIoU_B_before[0],
                 per_category_mIoU_B[0],
                per_category_mIoU_A[0],per_category_avg_learning_acc[0],
                per_category_forgetting[0]])

for cat_id in range(1,12):
    data.append([ZEISS_CATEGORIES[cat_id],per_category_mIoU_A_before[cat_id],
                 per_category_mIoU_B_before[cat_id],
                 per_category_mIoU_B[cat_id],
                per_category_mIoU_A[cat_id],per_category_avg_learning_acc[cat_id],
                per_category_forgetting[cat_id]])
    
    
table = wandb.Table(columns=columns, data=data)
wandb.log({"per_category_metrics_table": table})

print("**** Overall mIoU ****")
print(f"mIoU on task A before training on B: {mIoU_A_before}")
print(f"mIoU on task B before training on B: {mIoU_B_before}")
print("\n")
print(f"mIoU on task B after training on B: {mIoU_B}")
print(f"mIoU on task A after training on B: {mIoU_forgetting_A}")

print("\n**** Per category mIoU ****")
print(f"Per category mIoU on task A before training on B: {per_category_mIoU_A_before}")
print(f"Per category mIoU on task B before training on B: {per_category_mIoU_B_before}")
print("\n")
print(f"Per category mIoU on task B after training on B: {per_category_mIoU_B}")
print(f"Per category mIoU on task A after training on B: {per_category_mIoU_A}")

print("\n**** Average learning accuracies ****")
print(f"Average learning acc.: {avg_learning_acc}")
print(f"Per category Average learning acc.: {per_category_avg_learning_acc}")

print("\n**** Forgetting ****")
print(f"Total forgetting: {total_forgetting}")
print(f"Per category forgetting: {per_category_forgetting}")
wandb.finish()

if os.path.exists("artifacts/"):
    shutil.rmtree("artifacts/")

