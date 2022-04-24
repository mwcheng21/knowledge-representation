from cgi import test
from transformers import TrainingArguments
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import PLBartModel, PLBartConfig
from transformers import Trainer
import numpy as np
from datasets import load_metric
import os
from itertools import zip_longest
from datasets import Dataset, DatasetDict
from torch import tensor, squeeze


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
            self.model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-base")
        else:
            self.model = PLBartForConditionalGeneration.from_pretrained(pretrained_model_path, local_files_only=True)

        self.metric = load_metric("accuracy")

    def load_datasets(self, save_dir):
        '''Load datasets from <save_dir> files (3 modalities)'''
        train, eval, test  = self.combine_modalities(save_dir)

        self.train_dataset = train.map(self.tokenize_function, remove_columns=["text"])
        self.eval_dataset = eval.map(self.tokenize_function, remove_columns=["text"])
        self.test_dataset = test.map(self.tokenize_function, remove_columns=["text"])

        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
        model_inputs["labels"] = squeeze(self.tokenizer(examples["labels"], padding="max_length", return_tensors="pt").input_ids) #TODO: do I tokenize this?
        model_inputs["attention_mask"] = squeeze(model_inputs["attention_mask"])
        model_inputs["input_ids"] = squeeze(model_inputs["input_ids"])

        return model_inputs

    def combine_modalities(self, save_dir):
        file_names  = ['data.buggy_only', 'data.commit_msg', 'data.full_code_fullGraph', 'data.fixed_only']
        out = {}
        for mode in ["train", "test", "eval"]:
            data = []
            labels = [] 
            files = [open(os.path.join(save_dir, mode, x), encoding="utf-8") for x in file_names]
            for lines in zip_longest(*files):
                input = ""
                for i in range(len(lines)-1):
                    input = input + lines[i] + " <SEP> "
                input = input[:-7]
                labels.append(lines[-1])
                data.append(input)
            out[mode] = Dataset.from_dict({
                "text": data,
                "labels": labels
            })
        return out["train"], out["eval"], out["test"]

    def train(self):
        '''Finetune model using transformers Trainer class. Save final model in /models/model_name'''
        #initialize the transformers trainer
        # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
        trainer = Trainer(
            model=self.model,
            # args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        self.model.save_pretrained("model/" + self.model_name + "_finetuned.pt")

    def evaluate(self):
        '''Evaluate model on test set by accuracy'''
        
        preds = self.model(self.test_dataset)
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
        acc = model.evaluate()
        print("Accuracy: ", acc)

if __name__ == "__main__":
    model = Model("test")
    model.run("./data/single")