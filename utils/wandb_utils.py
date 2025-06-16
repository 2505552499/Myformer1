"""
Weights & Biases utilities for checkpoint management and logging.
"""

import os
import torch


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    """Save training checkpoint with all necessary information."""
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def save_checkpoint_to_wandb(checkpoint_path, epoch, mpjpe, is_best=False, use_wandb=False):
    """Save checkpoint to wandb as artifact"""
    if not use_wandb:
        return

    try:
        import wandb

        # Create artifact name based on checkpoint type
        if is_best:
            artifact_name = f"best_model_epoch_{epoch}"
            artifact_description = f"Best model at epoch {epoch} with MPJPE: {mpjpe:.4f}mm"
        else:
            artifact_name = f"latest_model_epoch_{epoch}"
            artifact_description = f"Latest model at epoch {epoch} with MPJPE: {mpjpe:.4f}mm"

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=artifact_description,
            metadata={
                "epoch": epoch,
                "mpjpe": mpjpe,
                "is_best": is_best,
                "framework": "pytorch"
            }
        )

        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        print(f"[INFO] Checkpoint uploaded to wandb: {artifact_name}")

    except Exception as e:
        print(f"[WARN] Failed to upload checkpoint to wandb: {e}")


def download_checkpoint_from_wandb(artifact_name, checkpoint_dir="checkpoint"):
    """Download checkpoint from wandb artifact without creating a new run"""
    try:
        import wandb
        import glob
        import shutil

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Check if checkpoint already exists locally
        existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth.tr"))
        if existing_checkpoints:
            print(f"[INFO] Found existing checkpoint: {existing_checkpoints[0]}")
            print(f"[INFO] Skipping download from wandb artifact: {artifact_name}")
            return existing_checkpoints[0]

        print(f"[INFO] No local checkpoint found, downloading from wandb artifact: {artifact_name}")

        # Use wandb API to download artifact without creating a run
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download()

        # Find the checkpoint file in the artifact
        checkpoint_files = glob.glob(os.path.join(artifact_dir, "*.pth.tr"))

        if not checkpoint_files:
            raise FileNotFoundError("No .pth.tr files found in the artifact")

        # Copy the checkpoint file to the checkpoint directory
        checkpoint_file = checkpoint_files[0]
        filename = os.path.basename(checkpoint_file)
        destination = os.path.join(checkpoint_dir, filename)

        shutil.copy(checkpoint_file, destination)

        print(f"[INFO] Checkpoint downloaded successfully: {destination}")
        return destination

    except Exception as e:
        print(f"[ERROR] Failed to download checkpoint from wandb: {e}")
        return None


def init_wandb_for_resume(wandb_id, project_name="MemoryInducedTransformer", args=None):
    """Initialize wandb for resuming training"""
    try:
        import wandb
        import pkg_resources

        wandb.init(
            id=wandb_id,
            project=project_name,
            resume="allow",
            settings=wandb.Settings(start_method='fork')
        )
        
        # Update config - allow value changes for resume
        if args:
            wandb.config.update({"run_id": wandb_id}, allow_val_change=True)
            wandb.config.update(args, allow_val_change=True)
            installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
            wandb.config.update({'installed_packages': installed_packages}, allow_val_change=True)
            
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize wandb for resume: {e}")
        return False


def init_wandb_for_new_run(wandb_id, wandb_name=None, project_name="MemoryInducedTransformer", args=None):
    """Initialize wandb for new training run"""
    try:
        import wandb
        import pkg_resources

        wandb.init(
            id=wandb_id,
            name=wandb_name,
            project=project_name,
            settings=wandb.Settings(start_method='fork')
        )
        
        # Update config
        if args:
            wandb.config.update({"run_id": wandb_id})
            wandb.config.update(args)
            installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
            wandb.config.update({'installed_packages': installed_packages})
            
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize wandb for new run: {e}")
        return False
