from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
import numpy as np
import os
from itertools import zip_longest
from datasets import Dataset
from torch import squeeze


class Model():
    def __init__(self, model_name, pretrained_tokenizer_path=None, pretrained_model_path=None):

        '''Config'''
        os.environ["CUDA_VISIBLE_DEVICES"]="1"

        '''Init function'''
        self.model_name = model_name
        #init pretrained model and tokenizer
        if pretrained_model_path==None:
            self.tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base", tgt_lang="java")
        else:
            self.tokenizer = PLBartTokenizer.from_pretrained(pretrained_tokenizer_path, local_files_only=True)
        
        #TODO: what model to use PLBartModel or PLBartForConditionalGeneration? Not sure the difference
        if pretrained_model_path==None:
            # configuration = PLBartConfig(**default_config)
            # print(configuration)
            # self.model = PLBartForConditionalGeneration(configuration)
            self.model = PLBartForConditionalGeneration.from_pretrained('uclanlp/plbart-base')
        else:
            self.model = PLBartForConditionalGeneration.from_pretrained(pretrained_model_path, local_files_only=True)
    
        


    def load_datasets(self, save_dir):
        '''Load datasets from <save_dir> files (3 modalities)'''
        train, eval, test  = self.combine_modalities(save_dir)

        self.train_dataset = train.map(self.tokenize_function, remove_columns=["text"])\
                                  .filter(lambda example: len(example["input_ids"]) <= 1024)
                                  
        self.eval_dataset = eval.map(self.tokenize_function, remove_columns=["text"])\
                                .filter(lambda example: len(example["input_ids"]) <= 1024)

        self.test_dataset = test.map(self.tokenize_function, remove_columns=["text"])\
                                .filter(lambda example: len(example["input_ids"]) <= 1024)

        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])



    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples["text"], padding="max_length", truncation=False, return_tensors="pt")
        model_inputs["labels"] = squeeze(self.tokenizer(examples["labels"], padding="max_length", return_tensors="pt").input_ids) #TODO: do I tokenize this?
        model_inputs["attention_mask"] = squeeze(model_inputs["attention_mask"])
        model_inputs["input_ids"] = squeeze(model_inputs["input_ids"])

        return model_inputs


    def combine_modalities(self, save_dir):
        file_names  = ['data.buggy_only', 'data.commit_msg', 'data.full_code_leaveOnly', 'data.fixed_only']
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
        config = {
            "output_dir": "test_trainer",
            "evaluation_strategy": "epoch",
            "generation_max_length": 512,
            # "per_gpu_train_batch_size": 2,
            # "per_device_eval_batch_size": 4
        }
        training_args = Seq2SeqTrainingArguments(**config)
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

        self.model.save_pretrained("model/" + self.model_name + "_finetuned.pt")

    def evaluate(self):
        '''Evaluate model on test set by accuracy'''
        test_pred = self.trainer.predict(self.test_dataset)
        return self.compute_metrics(test_pred)


    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels


    def acc_metric(self, preds, labels): 
        accur_count = 0
        sample_size = len(labels)
        assert len(preds) == len(labels)
        for i in range(len(preds)):
            accur_count += 1 if preds[i] == labels[i]  else 0
        
        return accur_count / sample_size * 100


    def compute_metrics(self, eval_pred):
        '''Helper function for what metric training should use'''
        labels = eval_pred.label_ids
        
        predictions = np.argmax(eval_pred.predictions[0], axis=-1)
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
  
        acc = self.acc_metric(decoded_preds, decoded_labels)

        return { 'Acc': round(acc, 4)}


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
    model.run("./data/medium")