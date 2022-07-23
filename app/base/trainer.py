## tool
import tqdm
import numpy as np
from seqeval.metrics import f1_score, accuracy_score
import time

## Bert
from transformers import (
    BertForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

## torch
import torch


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tag2idx = {"O": 0, "name": 1, "PAD": 2}
        self.tag_values = ["O", "name", "PAD"]

        print(f"device:{self.device}")

    def evaluation(self, test=False):
        self.model.eval()
        correct = 0
        total_loss = 0
        auc = 0
        predictions, true_labels = [], []

        with torch.no_grad():
            for step, batch in enumerate(tqdm.tqdm(self.val_dataloader), 0):

                if step > 0 and test:
                    break

                b_input_ids = batch["words"].to(self.device)
                b_input_mask = batch["mask"].to(self.device)
                b_labels = batch["tag"].to(self.device)

                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                # Move logits and labels to CPU
                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                # Calculate the accuracy for this batch of test sentences.
                total_loss += outputs[0].mean().item()

                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)

            pred_tags = [
                self.tag_values[p_i]
                for p, l in zip(predictions, true_labels)
                for p_i, l_i in zip(p, l)
                if self.tag_values[l_i] != "PAD"
            ]

            valid_tags = [
                self.tag_values[l_i]
                for l in true_labels
                for l_i in l
                if self.tag_values[l_i] != "PAD"
            ]

        return (
            accuracy_score(pred_tags, valid_tags),
            total_loss / len(self.val_dataloader),
        )

    def training_process(
        self,
        early_stopping=True,
        n_iter_no_change=5,
        max_epoch=10,
        save_params=False,
        verbose=True,
        learning_rate=1e-5,
        save_paths="model-best.pth",
    ):

        self.loss_values = []
        self.validation_loss_values = []

        self.FULL_FINETUNING = True
        self.max_grad_norm = 1.0

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_dataloader) * max_epoch
        lossmin = 100000
        earlystop = 0
        since = time.time()

        ##
        if self.FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "gamma", "beta"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay_rate": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay_rate": 0.0,
                },
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        ## Optimizer
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

        ## Create the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        for epoch in range(0, max_epoch):  # loop over the dataset multiple times
            total_loss = 0

            time.sleep(1)

            #####################
            ## Training phrase ##
            #####################

            self.model.train()
            for step, batch in enumerate(tqdm.tqdm(self.train_dataloader), 0):

                b_input_ids = batch["words"].to(self.device)
                b_input_mask = batch["mask"].to(self.device)
                b_labels = batch["tag"].to(self.device)

                self.model.zero_grad()

                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                # backward
                loss = outputs[0]
                loss.backward()
                total_loss += outputs[0].item()

                # Clip the norm of the gradient
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.max_grad_norm
                )

                self.optimizer.step()  # update parameters
                self.scheduler.step()  # Update the learning rate.

            avg_train_loss = total_loss / len(self.train_dataloader)

            #######################
            ## Evaluation phrase ##
            #######################

            self.model.eval()
            acc, avg_valid_loss = self.evaluation()

            ## Save loss
            self.loss_values.append(avg_train_loss)
            self.validation_loss_values.append(avg_valid_loss)

            ## verbose
            time_elapsed = time.time() - since

            if lossmin > avg_valid_loss:

                lossmin = avg_valid_loss
                earlystop = 0

                if save_params:
                    torch.save(self.model.state_dict(), save_paths)
                if verbose:
                    print(
                        f" time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | epoch:{epoch} train loss: {avg_train_loss:.4f} | validation loss: {avg_valid_loss:.4f} acc: {acc:.4f}",
                        end="",
                    )
            else:
                earlystop += 1
                if verbose:
                    print(
                        f" time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | epoch:{epoch} train loss: {avg_train_loss:.4f} | validation loss: {avg_valid_loss:.4f} acc: {acc:.4f} -- {earlystop}",
                        end="",
                    )

            if earlystop >= n_iter_no_change and early_stopping:
                if verbose:
                    print("\n earlystop")
                break

        print("\n Finished Training")
