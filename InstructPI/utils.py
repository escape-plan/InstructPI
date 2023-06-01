import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)
import pdb

class T5Generator:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample['pair'], max_length=512, truncation=True)
        # pdb.set_trace()
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        #Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["dev"],
            # eval_dataset=tokenized_datasets["test"] if tokenized_datasets.get("test") is not None else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer


    def get_labels(self, tokenized_dataset, batch_size = 4, max_length = 512, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length = max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    # def get_metrics(self, y_true, y_pred):
    #     total_pred = 0
    #     total_gt = 0
    #     tp = 0
    #     for gt, pred in zip(y_true, y_pred):
    #         gt_list = gt.split(', ')
    #         pred_list = pred.split(', ')
    #         total_pred+=len(pred_list)
    #         total_gt+=len(gt_list)
    #         for gt_val in gt_list:
    #             for pred_val in pred_list:
    #                 if pred_val.lower() == gt_val.lower() or gt_val.lower() == pred_val.lower():
    #                     tp+=1
    #                     break
    #     p = tp/total_pred
    #     r = tp/total_gt
    #     return p, r, 2*p*r/(p+r), None

    def get_metrics(self, y_true, y_pred):
        total_num = 0
        correct = 0

        for pred, truth in zip(y_pred, y_true):
            total_num += 1
            if truth.lower() == pred.lower():
                correct += 1

        return 1.0*correct/total_num