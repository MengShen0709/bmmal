import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import ClasswiseWrapper, MetricCollection, CalibrationError
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, \
    MulticlassAUROC
from torchmetrics import Metric


class Contribution(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("contribution", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, contribution: torch.Tensor):
        self.contribution += torch.sum(contribution)
        self.total += contribution.size(0)

    def compute(self):
        return self.contribution.float() / self.total


class MMClassification(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 config: dict,
                 exp_params: dict,
                 tb_logger: TensorBoardLogger,
                 mc_iteration: int = 0
                 ) -> None:
        super(MMClassification, self).__init__()

        self.automatic_optimization = True
        self.model = model
        self.curr_device = None
        self.exp_params = exp_params
        self.mc_iteration = mc_iteration
        self.mc_iteration_m1 = 0
        self.mc_iteration_m2 = 0
        torch.autograd.set_detect_anomaly(True)
        self.config = config
        self.num_classes = self.config['model_params']['class_num']
        self.m1 = self.config['m1']
        self.m2 = self.config['m2']

        self.loss_function = nn.CrossEntropyLoss()

        self.m1_metrics = MetricCollection(
            {"Top-1": MulticlassAccuracy(num_classes=self.num_classes),
             "F1Score": MulticlassF1Score(num_classes=self.num_classes),
             "Precision": MulticlassPrecision(num_classes=self.num_classes),
             "Recall": MulticlassRecall(num_classes=self.num_classes),
             "AUROC": MulticlassAUROC(num_classes=self.num_classes),
             "ECE": CalibrationError(n_bins=32, norm='l1', task='multiclass', num_classes=self.num_classes),
             "accuracy": ClasswiseWrapper(MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average=None),
                                          labels=None)},
            prefix=f'{self.m1}/'
        )
        self.m2_metrics = self.m1_metrics.clone(prefix=f'{self.m2}/')

        self.early_stopping_monitor_mm = MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average='macro')
        self.early_stopping_monitor_m1 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average='macro')
        self.early_stopping_monitor_m2 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1, average='macro')
        self.mm_metrics = self.m1_metrics.clone(prefix=f'MM/')

        self.tb_logger = tb_logger

        self.log_dir = self.tb_logger.log_dir.replace(self.tb_logger.sub_dir, "")

        self.m1_contribution = MetricCollection(
            {"Contribution": Contribution()},
            prefix=f'{self.m1}/'
        )
        self.m2_contribution = self.m1_contribution.clone(prefix=f"{self.m2}/")

        self.m1_delta = MetricCollection(
            {"Delta": Contribution()},
            prefix=f'{self.m1}/'
        )
        self.m2_delta = self.m1_delta.clone(prefix=f"{self.m2}/")

        self.m1_weight = 1 / 3
        self.m2_weight = 1 / 3
        self.mm_weight = 1 / 3

        self.mix = False
        self.suppress = False
        self.integrated_gradient = False
        self.avg_m1_alpha = 0
        self.avg_m2_alpha = 0

        self.validation_step_outputs = []

    @property
    def dimension_m1(self):
        return self.model.m1_model.fc.weight.flatten().size(0) + self.model.m1_model.fc.bias.flatten().size(0)

    @property
    def dimension_m2(self):
        return self.model.m2_model.fc.weight.flatten().size(0) + self.model.m2_model.fc.bias.flatten().size(0)

    @property
    def global_relative_step(self):
        return int(self.global_step / self.trainer.estimated_stepping_batches * 100000)

    def forward(self, m1, m2):
        pass

    def on_train_batch_start(self, batch, batch_idx):
        # adjust learning rate for warm up
        opt = self.optimizers()
        warm_up_epoch = self.exp_params["warm_up_epoch"]
        warm_up_step = self.trainer.estimated_stepping_batches // self.trainer.max_epochs * warm_up_epoch

        if self.global_step < warm_up_step:

            # linear learning rate warm up
            lr_scale = (self.global_step + 1) / warm_up_step
            for i in range(len(opt.param_groups)):
                opt.param_groups[i]['lr'] = lr_scale * self.initial_lr[i]

    def on_train_start(self) -> None:
        self.log("Train/Top-1/MM", 0., sync_dist=True, on_epoch=True, prog_bar=True)
        self.log(f"Train/Top-1/{self.m1}", 0., sync_dist=True, on_epoch=True, prog_bar=False)
        self.log(f"Train/Top-1/{self.m2}", 0., sync_dist=True, on_epoch=True, prog_bar=False)
        self.log("Train/Top-1", 0., sync_dist=True, on_epoch=True)

    def training_step(self, batch, batch_idx):

        m1_input, m2_input, labels = batch

        batch_size = int(labels.size(0))
        mm_labels = labels.clone()
        m1_labels = labels.clone()
        m2_labels = labels.clone()

        self.curr_device = mm_labels.device

        m1_feature, m2_feature = self.model.feature_extraction(m1_input, m2_input)
        m1_logits, m2_logits, mm_logits = self.model(m1_feature, m2_feature)

        # for early stopping

        self.early_stopping_monitor_mm.update(torch.softmax(mm_logits, dim=-1), mm_labels)
        self.early_stopping_monitor_m1.update(torch.softmax(m1_logits, dim=-1), m1_labels)
        self.early_stopping_monitor_m2.update(torch.softmax(m2_logits, dim=-1), m2_labels)

        m1_loss = self.loss_function(m1_logits, m1_labels)
        m2_loss = self.loss_function(m2_logits, m2_labels)
        mm_loss = self.loss_function(mm_logits, mm_labels)

        self.tb_logger.experiment.add_scalar(f"Train/{self.m1}_loss", m1_loss, self.global_relative_step)
        self.tb_logger.experiment.add_scalar(f"Train/{self.m2}_loss", m2_loss, self.global_relative_step)
        self.tb_logger.experiment.add_scalar("Train/mm_loss", mm_loss, self.global_relative_step)

        return self.mm_weight * mm_loss + self.m1_weight * m1_loss + self.m2_weight * m2_loss

    def on_train_epoch_end(self) -> None:
        # for early stopping
        self.log("Train/Top-1/MM", self.early_stopping_monitor_mm.compute(), sync_dist=True, on_epoch=True,
                 prog_bar=True)
        self.log(f"Train/Top-1/{self.m1}", self.early_stopping_monitor_m1.compute(), sync_dist=True, on_epoch=True,
                 prog_bar=False)
        self.log(f"Train/Top-1/{self.m2}", self.early_stopping_monitor_m2.compute(), sync_dist=True, on_epoch=True,
                 prog_bar=False)
        self.log("Train/Top-1",
                 (self.early_stopping_monitor_mm.compute() +
                  self.early_stopping_monitor_m1.compute() +
                  self.early_stopping_monitor_m2.compute()) / 3,
                 sync_dist=True, on_epoch=True)
        self.early_stopping_monitor_mm.reset()
        self.early_stopping_monitor_m1.reset()
        self.early_stopping_monitor_m2.reset()

    def validation_step(self, batch, batch_idx):
        m1_input, m2_input, labels = batch

        batch_size = int(labels.size(0))
        mm_labels = labels.clone()
        m1_labels = labels.clone()
        m2_labels = labels.clone()

        self.curr_device = mm_labels.device
        m1_feature, m2_feature = self.model.feature_extraction(m1_input, m2_input)
        m1_logits, m2_logits, mm_logits = self.model(m1_feature, m2_feature)

        m1_loss = self.loss_function(m1_logits, m1_labels)
        m2_loss = self.loss_function(m2_logits, m2_labels)
        mm_loss = self.loss_function(mm_logits, mm_labels)

        self.mm_metrics.update(torch.softmax(mm_logits, dim=-1), mm_labels)
        self.m1_metrics.update(torch.softmax(m1_logits, dim=-1), m1_labels)
        self.m2_metrics.update(torch.softmax(m2_logits, dim=-1), m2_labels)
        self.validation_step_outputs.append([m1_loss, m2_loss, mm_loss])
        return m1_loss, m2_loss, mm_loss

    def on_validation_epoch_end(self):
        mm_loss = 0.
        m1_loss = 0.
        m2_loss = 0.
        i = 0
        for i, (each_m1_loss, each_m2_loss, each_mm_loss) in enumerate(self.validation_step_outputs):
            m1_loss += float(each_m1_loss)
            m2_loss += float(each_m2_loss)
            mm_loss += float(each_mm_loss)
        mm_loss /= i
        m1_loss /= i
        m2_loss /= i
        self.tb_logger.experiment.add_scalar("Val/mm_loss", mm_loss, self.global_relative_step)
        self.tb_logger.experiment.add_scalar(f"Val/{self.m1}_loss", m1_loss, self.global_relative_step)
        self.tb_logger.experiment.add_scalar(f"Val/{self.m2}_loss", m2_loss, self.global_relative_step)
        self.log_dict(self.mm_metrics.compute(), sync_dist=True)
        self.log_dict(self.m1_metrics.compute(), sync_dist=True)
        self.log_dict(self.m2_metrics.compute(), sync_dist=True)
        self.mm_metrics.reset()
        self.m1_metrics.reset()
        self.m2_metrics.reset()

        self.validation_step_outputs.clear()

    def on_test_start(self) -> None:
        self.mm_metrics.reset()
        self.m1_metrics.reset()
        self.m2_metrics.reset()

    def test_step(self, batch, batch_idx):
        m1_input, m2_input, labels = batch

        batch_size = int(labels.size(0))
        mm_labels = labels.clone()
        m1_labels = labels.clone()
        m2_labels = labels.clone()

        self.curr_device = mm_labels.device
        m1_feature, m2_feature = self.model.feature_extraction(m1_input, m2_input)
        m1_logits, m2_logits, mm_logits = self.model(m1_feature, m2_feature)

        mm_prob = torch.softmax(mm_logits, dim=-1)
        m1_prob = torch.softmax(m1_logits, dim=-1)
        m2_prob = torch.softmax(m2_logits, dim=-1)

        self.mm_metrics.update(mm_prob, mm_labels)
        self.m1_metrics.update(m1_prob, m1_labels)
        self.m2_metrics.update(m2_prob, m2_labels)

        if isinstance(m1_feature, list):
            # For NL-Gate Fusion
            _, _, mm_logits_drop_m1 = self.model([torch.zeros_like(f) for f in m1_feature], m2_feature)
            _, _, mm_logits_drop_m2 = self.model(m1_feature, [torch.zeros_like(f) for f in m2_feature])
            _, _, mm_logits_drop_m1_m2 = self.model([torch.zeros_like(f) for f in m1_feature],
                                                    [torch.zeros_like(f) for f in m2_feature])
        else:
            _, _, mm_logits_drop_m1 = self.model(torch.zeros_like(m1_feature), m2_feature)
            _, _, mm_logits_drop_m2 = self.model(m1_feature, torch.zeros_like(m2_feature))
            _, _, mm_logits_drop_m1_m2 = self.model(torch.zeros_like(m1_feature), torch.zeros_like(m2_feature))

        mm_probs_m1_drop = torch.softmax(mm_logits_drop_m1, dim=-1)
        mm_probs_m2_drop = torch.softmax(mm_logits_drop_m2, dim=-1)
        mm_probs_m1_m2_drop = torch.softmax(mm_logits_drop_m1_m2, dim=-1)

        # if delta_m1 is small, meaning that m1 contributes less
        delta_m1 = (mm_prob - mm_probs_m1_drop + mm_probs_m2_drop - mm_probs_m1_m2_drop) / 2
        delta_m2 = (mm_prob - mm_probs_m2_drop + mm_probs_m1_drop - mm_probs_m1_m2_drop) / 2

        contribution_m1 = torch.abs(delta_m1) / (torch.abs(delta_m1) + torch.abs(delta_m2))
        contribution_m2 = torch.abs(delta_m2) / (torch.abs(delta_m1) + torch.abs(delta_m2))

        self.m1_contribution.update(contribution_m1.gather(1, mm_labels.unsqueeze(1)))
        self.m2_contribution.update(contribution_m2.gather(1, mm_labels.unsqueeze(1)))

        self.m1_delta.update(delta_m1.gather(1, mm_labels.unsqueeze(1)))
        self.m2_delta.update(delta_m2.gather(1, mm_labels.unsqueeze(1)))

    def on_test_epoch_end(self):
        self.log_dict(self.mm_metrics.compute(), sync_dist=True)
        self.log_dict(self.m1_metrics.compute(), sync_dist=True)
        self.log_dict(self.m2_metrics.compute(), sync_dist=True)
        self.log_dict(self.m1_contribution.compute(), sync_dist=True)
        self.log_dict(self.m2_contribution.compute(), sync_dist=True)
        self.log_dict(self.m1_delta.compute(), sync_dist=True)
        self.log_dict(self.m2_delta.compute(), sync_dist=True)

        self.mm_metrics.reset()
        self.m1_metrics.reset()
        self.m2_metrics.reset()
        self.m1_contribution.reset()
        self.m2_contribution.reset()
        self.m1_delta.reset()
        self.m2_delta.reset()

    def predict_step(self, batch, batch_idx):
        m1_input, m2_input, labels = batch

        batch_size = int(labels.size(0))
        mm_labels = labels.clone()
        m1_labels = labels.clone()
        m2_labels = labels.clone()

        self.curr_device = mm_labels.device
        m1_feature, m2_feature = self.model.feature_extraction(m1_input, m2_input)

        if self.mc_iteration > 0:

            m1_logits, m2_logits, mm_logits, m1_z, m2_z, mm_z = self.model(m1_feature, m2_feature,
                                                                           return_latent=True)
            for name, module in self.model.named_children():
                if module.__class__.__name__ == "Dropout":
                    module.train()

            mm_logits = []
            m1_logits = []
            m2_logits = []
            for i in range(self.mc_iteration):
                each_m1_logits, each_m2_logits, each_mm_logits, each_m1_z, each_m2_z, each_mm_z \
                    = self.model(m1_feature, m2_feature, return_latent=True)
                mm_logits.append(each_mm_logits.unsqueeze(0))
                m1_logits.append(each_m1_logits.unsqueeze(0))
                m2_logits.append(each_m2_logits.unsqueeze(0))
            mm_logits = torch.cat(mm_logits, dim=0)
            m1_logits = torch.cat(m1_logits, dim=0)
            m2_logits = torch.cat(m2_logits, dim=0)
            m1_z = None
            m2_z = None
            mm_z = None

            mm_probs = torch.softmax(mm_logits, dim=-1)
            m1_probs = torch.softmax(m1_logits, dim=-1)
            m2_probs = torch.softmax(m2_logits, dim=-1)

            contribution_m1, contribution_m2, delta_m1, delta_m2, delta_m1_logits, delta_m2_logits = [None] * 6

        else:
            m1_logits, m2_logits, mm_logits, m1_z, m2_z, mm_z = self.model(m1_feature, m2_feature,
                                                                           return_latent=True)

            mm_probs = torch.softmax(mm_logits, dim=-1)
            m1_probs = torch.softmax(m1_logits, dim=-1)
            m2_probs = torch.softmax(m2_logits, dim=-1)

            if isinstance(m1_feature, list):
                # For NL-Gate Fusion
                _, _, mm_logits_drop_m1 = self.model([torch.zeros_like(f) for f in m1_feature], m2_feature)
                _, _, mm_logits_drop_m2 = self.model(m1_feature, [torch.zeros_like(f) for f in m2_feature])
                _, _, mm_logits_drop_m1_m2 = self.model([torch.zeros_like(f) for f in m1_feature],
                                                        [torch.zeros_like(f) for f in m2_feature])
            else:
                _, _, mm_logits_drop_m1 = self.model(torch.zeros_like(m1_feature), m2_feature)
                _, _, mm_logits_drop_m2 = self.model(m1_feature, torch.zeros_like(m2_feature))
                _, _, mm_logits_drop_m1_m2 = self.model(torch.zeros_like(m1_feature), torch.zeros_like(m2_feature))

            mm_probs_m1_drop = torch.softmax(mm_logits_drop_m1, dim=-1)
            mm_probs_m2_drop = torch.softmax(mm_logits_drop_m2, dim=-1)
            mm_probs_m1_m2_drop = torch.softmax(mm_logits_drop_m1_m2, dim=-1)

            # if delta_m1 is small, meaning that m1 contributes less
            delta_m1 = (mm_probs - mm_probs_m1_drop + mm_probs_m2_drop - mm_probs_m1_m2_drop) / 2
            delta_m2 = (mm_probs - mm_probs_m2_drop + mm_probs_m1_drop - mm_probs_m1_m2_drop) / 2

            delta_m1_logits = (mm_logits - mm_logits_drop_m1 + mm_logits_drop_m2 - mm_logits_drop_m1_m2) / 2
            delta_m2_logits = (mm_logits - mm_logits_drop_m2 + mm_logits_drop_m1 - mm_logits_drop_m1_m2) / 2

            contribution_m1 = torch.abs(delta_m1) / (torch.abs(delta_m1) + torch.abs(delta_m2))
            contribution_m2 = torch.abs(delta_m2) / (torch.abs(delta_m1) + torch.abs(delta_m2))

        return m1_probs, m2_probs, mm_probs, m1_z, m2_z, mm_z, contribution_m1, contribution_m2, delta_m1, delta_m2, delta_m1_logits, delta_m2_logits

    def configure_optimizers(self):

        for p in self.model.parameters():
            p.requires_grad = True

        optimizer_dict = {
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
        }

        if self.model.__class__.__name__ == "AVModelNLGate":
            print("Detecting AVModelNLGate...")
            optimizer = optimizer_dict[self.exp_params["optimizer"]]([
                {"params": [p for p in self.model.mm_classifier.parameters() if p.requires_grad] +
                           [p for p in self.model.m1_classifier.parameters() if p.requires_grad] +
                           [p for p in self.model.m2_classifier.parameters() if p.requires_grad],
                 **self.exp_params["classifier"]},
                {"params": [p for p in self.model.m1_model.parameters() if p.requires_grad],
                 **self.exp_params["m1_model"]},
                {"params": [p for p in self.model.m2_model.parameters() if p.requires_grad],
                 **self.exp_params["m2_model"]},
                {"params": [p for p in self.model.nl_gate.parameters() if p.requires_grad] +
                           [p for p in self.model.layer4.parameters() if p.requires_grad],
                 **self.exp_params["nl_gate"]}
            ])
        else:
            optimizer = optimizer_dict[self.exp_params["optimizer"]]([
                {"params": [p for p in self.model.mm_classifier.parameters() if p.requires_grad] +
                           [p for p in self.model.m1_classifier.parameters() if p.requires_grad] +
                           [p for p in self.model.m2_classifier.parameters() if p.requires_grad],
                 **self.exp_params["classifier"]},
                {"params": [p for p in self.model.m1_model.parameters() if p.requires_grad],
                 **self.exp_params["m1_model"]},
                {"params": [p for p in self.model.m2_model.parameters() if p.requires_grad],
                 **self.exp_params["m2_model"]},
            ])

        self.initial_lr = [p_g['lr'] for p_g in optimizer.param_groups]
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                           milestones=self.exp_params["milestones"], gamma=0.1)

        return [optimizer], [lr_schedule]
