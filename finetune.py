from cgi import test
from transformers import TrainingArguments
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import PLBartModel, PLBartConfig
from transformers import Trainer
import numpy as np
from datasets import load_metric
import os
from itertools import zip_longest
from dataset import CodeDataset

class Model():
    def __init__(self, model_name, pretrained_tokenizer_path=None, pretrained_model_path=None):
        '''Init function'''
        self.model_name = model_name
        #init pretrained model and tokenizer
        if pretrained_model_path==None:
            self.tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base", tgt_lang="java")
        else:
            self.tokenizer = PLBartTokenizer.from_pretrained(pretrained_tokenizer_path, local_files_only=True)

        #TODO: what model to use PLBartModel or PLBartForConditionalGeneration? Not sure the difference
        if pretrained_model_path==None:
            self.model = PLBartModel.from_pretrained("uclanlp/plbart-base")
        else:
            self.model = PLBartModel.from_pretrained(pretrained_model_path, local_files_only=True)

        self.metric = load_metric("accuracy")

    def load_datasets(self, save_dir):
        '''Load datasets from <save_dir> files (3 modalities)'''

        train_text, train_labels = self.combine_modalities(save_dir, "train/")
        eval_text, eval_labels = self.combine_modalities(save_dir, "eval/")
        test_text, test_labels = self.combine_modalities(save_dir, "test/")

        padding = True
        truncation = True

        train_tokens = self.tokenizer(train_text, padding=padding, truncation=truncation, return_tensors="pt")
        eval_tokens = self.tokenizer(eval_text, padding=padding, truncation=truncation, return_tensors="pt")
        test_tokens = self.tokenizer(eval_text, padding=padding, truncation=truncation, return_tensors="pt")

        self.train_dataset = CodeDataset(train_tokens, train_labels)
        self.eval_dataset = CodeDataset(eval_tokens, eval_labels)
        self.test_dataset = CodeDataset(test_text, test_labels)

    def combine_modalities(self, save_dir, sub_dir):
        file_names  = ['data.buggy_only', 'data.commit_msg', 'data.full_code_fullGraph', 'data.fixed_only']

        data = []
        labels = [] 
        files = [open(os.path.join(save_dir, sub_dir + x), encoding="utf-8") for x in file_names]
        for lines in zip_longest(*files):
            input = ""
            for i in range(len(lines)-1):
                input = input + lines[i] + " <SEP> "
            input = input[:-7]
            labels.append(lines[-1])
            data.append(input)
        return data, labels

    def train(self):
        '''Finetune model using transformers Trainer class. Save final model in /models/model_name'''
        #initialize the transformers trainer
        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        self.model.save_pretrained("model/" + self.model_name + "_finetuned.pt")
        #TODO: Do we need to train tokenizer?
        self.tokenizer.save_pretrained("model/" + self.model_name +"_tokenizer_finetuned.pt")

    def evaluate(self):
        '''Evaluate model on test set by accuracy'''
        preds = self.model.predict(self.test_dataset.encodings)
        return self.metric.compute(predictions=preds, references=self.test_dataset.labels)

    def compute_metrics(self, eval_pred):
        '''Helper function for what metric training should use'''
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def run(self, folder_name):
        '''Run finetuning'''
        print("Loading datasets...")
        model.load_datasets(folder_name)
        print("Finetuning...")
        model.train()
        print("Evaluating...")
        bleuScore = model.evaluate()
        print("Bleu Score: ", bleuScore)

if __name__ == "__main__":
    model = Model("test")
    model.run("./data/medium")