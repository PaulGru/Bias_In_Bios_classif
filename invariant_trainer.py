import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import transformers
from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from transformers.trainer_callback import TrainerState
from transformers.utils import logging

from tqdm import tqdm

import wandb
import math
import os
import numpy as np
from itertools import cycle
import random
from torch.amp import autocast, GradScaler
from typing import Optional

logger = logging.get_logger(__name__)

def compute_moving_average(values, window_size=10):
        if len(values) < window_size:
            return sum(values) / max(len(values), 1)
        return sum(values[-window_size:]) / window_size

    
def ib_penalty(embeddings: torch.Tensor):
    """
    Computes variance-based Information Bottleneck penalty.
    Args:
        embeddings: tensor of shape (batch_size, hidden_dim)
    Returns:
        scalar tensor: sum of variances across dimensions
    """
    mean = embeddings.mean(dim=0)
    var = ((embeddings - mean) ** 2).mean(dim=0)
    return var.sum()


class InvariantTrainer(transformers.Trainer):

    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        optimizer, lr_scheduler = None, None
        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler
    

    def invariant_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            resume_from_checkpoint: Optional[str] = None,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            **kwargs,
    ):

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])
        
        steps_per_epoch = math.floor(
            min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        if nb_steps is not None:
            num_train_epochs = max(1, math.floor(nb_steps / steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = steps_per_epoch * num_train_epochs

        dataloaders, head_optimizers, head_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])
            head_optimizers[env_name], head_schedulers[env_name] = self.create_optimizer_and_scheduler(
                self.model.classifier_heads[env_name],
                num_training_steps=max_steps
            )

        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  steps_per_epoch = {steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        self.state.global_step = 0
        
        recent_losses = []

        self.scaler = GradScaler()

        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}

        pbar = tqdm(total=max_steps, desc="Training progress")
        while self.state.global_step < max_steps:
            
            for env_name in training_set.keys():

                logger.info(f" Update on environement {env_name}")
                phi_optimizer.zero_grad()
                head_optimizers[env_name].zero_grad()

                batch = next(iter_loaders[env_name])     
                
                self.model.train()
                batch = self._prepare_inputs(batch)

                if self.use_apex:
                    with autocast():
                        loss = self.compute_loss(self.model, batch)
                else:
                    loss = self.compute_loss(self.model, batch)

                if self.args.n_gpu > 1:
                    loss = loss.mean()
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                if self.use_apex:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss = loss.detach()
                
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    if self.use_apex:
                        self.scaler.unscale_(phi_optimizer)
                        self.scaler.unscale_(head_optimizers[env_name])

                    if hasattr(phi_optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        phi_optimizer.clip_grad_norm(self.args.max_grad_norm)
                        head_optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.max_grad_norm,
                        )

                if self.use_apex:
                    self.scaler.step(phi_optimizer)
                    self.scaler.step(head_optimizers[env_name])
                    self.scaler.update()
                else:
                    phi_optimizer.step()
                    head_optimizers[env_name].step()

                # Mise à jour des schedulers
                phi_scheduler.step()
                head_schedulers[env_name].step()

                recent_losses.append(loss.item())
                moving_avg_loss = compute_moving_average(recent_losses, window_size=20)

                if self.is_world_process_zero() and self.state.global_step % 100 == 0:
                        wandb.log({
                            "training/train_loss": loss.item(),
                            "training/train_loss_moving_avg": moving_avg_loss
                        }, step=self.state.global_step
                        )

                if saving_heads and self.state.global_step % nb_steps_heads_saving == 0:
                    self.save_heads(self.state.global_step)
                if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                    self.save_intermediary_model(self.state.global_step)

                if (
                    self.args.do_eval
                    and getattr(self.args, "evaluation_strategy", None) == "steps"
                    and getattr(self.args, "eval_steps", 0) > 0
                    and self.state.global_step % self.args.eval_steps == 0
                ):
                    metrics = self.evaluate()  # évalue sur self.eval_dataset
                    logger.info(f"*** Evaluation at step {self.state.global_step}: {metrics}")
                    # log vers WandB si processus principal
                    if self.is_world_process_zero() and wandb.run:
                        wandb.log(
                            {f"eval/{k}": v for k, v in metrics.items()},
                            step=self.state.global_step,
                        )

                self.state.global_step += 1
                pbar.update(1)
        
        pbar.close()              
        print("=== Entraînement du modèle terminé. Nombre total de steps :", self.state.global_step)
 

    def invariant_train_games(
        self,
        training_set,
        nb_steps: Optional[int] = None,
        nb_steps_heads_saving: Optional[int] = 0,
        resume_from_checkpoint: Optional[str] = None,
        num_train_epochs: Optional[int] = 1,
        nb_steps_model_saving: Optional[int] = 0,
        **kwargs,
    ):

        # Flag F-IRM : si défini, on ne met jamais à jour phi
        freeze_phi = getattr(self, 'freeze_phi', True) #True
        head_updates_per_encoder_update = getattr(self.args, "head_updates_per_encoder_update")
        print(f"head_updates_per_encoder_update: {head_updates_per_encoder_update}")

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        min_train_size = min(len(data["train"]) for _, data in training_set.items())
        
        steps_per_epoch = math.floor(
            min_train_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        if nb_steps is not None:
            num_train_epochs = max(1, math.floor(nb_steps / steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = steps_per_epoch * num_train_epochs

        dataloaders, head_optimizers, head_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])
            head_optimizers[env_name], head_schedulers[env_name] = self.create_optimizer_and_scheduler(
                self.model.classifier_heads[env_name],
                num_training_steps=max_steps
            )
        
        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        # Move model to device
        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  steps_per_epoch = {steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        self.state.global_step = 0

        recent_head_losses = []
        recent_phi_losses = []

        self.scaler = GradScaler()
        
        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}
        
        pbar = tqdm(total=max_steps, desc="Training progress")
        while self.state.global_step < max_steps:
            # === Phase 1: update heads ===
                
            for _ in range(head_updates_per_encoder_update):
                self.model.encoder.requires_grad_(False)
                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")
                    self.model.classifier_heads[env_name].requires_grad_(True)
                    head_optimizers[env_name].zero_grad()

                    batch = next(iter_loaders[env_name])
                    
                    self.model.train()
                    batch = self._prepare_inputs(batch)

                    if self.use_apex:
                        with autocast():
                            loss_head = self.compute_loss(self.model, batch)
                    else:
                        loss_head = self.compute_loss(self.model, batch)

                    if self.args.n_gpu > 1:
                        loss_head = loss_head.mean()
                
                    if self.args.gradient_accumulation_steps > 1:
                        loss_head = loss_head / self.args.gradient_accumulation_steps
                    
                    if self.use_apex:
                        self.scaler.scale(loss_head).backward()
                    else:
                        loss_head.backward()

                    loss_head = loss_head.detach()
                    recent_head_losses.append(loss_head.item())

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_apex:
                            self.scaler.unscale_(head_optimizers[env_name])
                        
                        if hasattr(head_optimizers[env_name], "clip_grad_norm"):
                            head_optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.classifier_heads[env_name].parameters(),
                                self.args.max_grad_norm,
                            )
                    
                    if self.use_apex:
                        self.scaler.step(head_optimizers[env_name])
                    else:
                        head_optimizers[env_name].step()

                    head_schedulers[env_name].step()

                    moving_avg_head = compute_moving_average(recent_head_losses, window_size=20)
                    if self.is_world_process_zero() and self.state.global_step % 100 == 0:
                        wandb.log({
                            "training/head_loss": loss_head.item(),
                            "training/head_loss_moving_avg": moving_avg_head,
                        }, step=self.state.global_step)

                    if nb_steps_heads_saving and self.state.global_step % nb_steps_heads_saving == 0:
                        self.save_heads(self.state.global_step)
                    if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                        self.save_intermediary_model(self.state.global_step)

                    # === Évaluation périodique tous les eval_steps ===
                    if (
                        self.args.do_eval
                        and getattr(self.args, "evaluation_strategy", None) == "steps"
                        and getattr(self.args, "eval_steps", 0) > 0
                        and self.state.global_step % self.args.eval_steps == 0
                    ):
                        metrics = self.evaluate()  # évalue sur `eval_dataset`
                        logger.info(f"*** Evaluation at step {self.state.global_step}: {metrics}")
                        # Log vers WandB si on est le processus principal
                        if self.is_world_process_zero() and wandb.run:
                            wandb.log(
                                {f"eval/{k}": v for k, v in metrics.items()},
                                step=self.state.global_step,
                            )

                    self.state.global_step += 1
                    pbar.update(1)

            # === Phase 2: update shared encoder ===
            if not freeze_phi:
                self.model.encoder.requires_grad_(True)
                for env_name in training_set.keys():
                    self.model.classifier_heads[env_name].requires_grad_(False)

                phi_optimizer.zero_grad()
                total_phi_loss = 0.0

                for env_name in training_set.keys():
                    batch = next(iter_loaders[env_name])
                    
                    self.model.train()
                    batch = self._prepare_inputs(batch)

                    if self.use_apex:
                        with autocast():
                            phi_loss = self.compute_loss(self.model, batch)
                    else:
                        phi_loss = self.compute_loss(self.model, batch)

                    if self.args.n_gpu > 1:
                        phi_loss = phi_loss.mean()
                
                    if self.args.gradient_accumulation_steps > 1:
                        phi_loss = phi_loss / self.args.gradient_accumulation_steps


                    if self.use_apex:
                            self.scaler.scale(phi_loss).backward()
                    else:
                        phi_loss.backward()

                    phi_loss = phi_loss.detach()
                    total_phi_loss += phi_loss.item()
                    
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_apex:
                            self.scaler.unscale_(phi_optimizer)
                        
                        if hasattr(phi_optimizer, "clip_grad_norm"):
                            phi_optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.encoder.parameters(),
                                self.args.max_grad_norm,
                            )
                        
                    if self.use_apex:
                        self.scaler.step(phi_optimizer)
                        self.scaler.update()
                    else:
                        phi_optimizer.step()

                    phi_scheduler.step()

                    recent_phi_losses.append(total_phi_loss)
                    moving_avg_phi = compute_moving_average(recent_phi_losses, window_size=20)
                    if self.is_world_process_zero() and self.state.global_step % 100 == 0:
                        wandb.log({
                            "training/phi_loss": total_phi_loss,
                            "training/phi_loss_moving_avg": moving_avg_phi,
                        }, step=self.state.global_step)

        pbar.close()                  
        print("=== Training complete. Total rounds:", self.state.global_step / len(training_set))


    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        # Ne sauvegarder que si ce processus est le principal
        if not self.is_world_process_zero():
            return
        
        print("saving-heads")
        if not os.path.exists("classifier_heads"):
            os.makedirs("classifier_heads")

        for env, classifier_heads in self.model.classifier_heads.items():
            filepath = os.path.join("classifier_heads", "{}-{}".format(env, step_count))
            
            if hasattr(classifier_heads, "dense"):
                np.save(filepath, classifier_heads.dense.weight.data.cpu().numpy())
            elif hasattr(classifier_heads, "decoder"):
                np.save(filepath, classifier_heads.decoder.weight.data.cpu().numpy())
            elif hasattr(classifier_heads, "vocab_projector"):
                np.save(filepath, classifier_heads.vocab_projector.weight.data.cpu().numpy())
            else:
                print(f"La tête pour l'environnement {env} ne possède pas d'attribut de sauvegarde connu.")


    def get_single_train_dataloader(self, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator
        )