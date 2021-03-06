from cgi import test
from tkinter.messagebox import NO
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import PLBartForConditionalGeneration, PLBartTokenizer, AdamW, get_constant_schedule
import numpy as np
from itertools import zip_longest
from datasets import Dataset
from torch import squeeze, cuda
import torch
import os
import math


class Model():
    def __init__(self, model_name, pretrained_tokenizer_path=None, pretrained_model_path=None):

        '''Config'''
        if(cuda.is_available()): print('GPU in using', cuda.current_device())

        self.batch_size = 1000
        self.MAX_EMBEDDINGS = 1024
        self.train_portions = []
        self.eval_portions = []

        '''Init function'''
        self.model_name = model_name

        #init pretrained model and tokenizer
        if pretrained_tokenizer_path==None:
            self.tokenizer = PLBartTokenizer.from_pretrained('uclanlp/plbart-base')
        else:
            self.tokenizer = PLBartTokenizer.from_pretrained(pretrained_tokenizer_path, local_files_only=True)
        
        #TODO: what model to use PLBartModel or PLBartForConditionalGeneration? Not sure the difference
        if pretrained_model_path==None:
            self.model = PLBartForConditionalGeneration.from_pretrained('uclanlp/plbart-base')
        else:
            print(pretrained_model_path)
            self.model = PLBartForConditionalGeneration.from_pretrained(pretrained_model_path, local_files_only=True)
    

    def load_datasets(self, save_dir):
        print(save_dir)
        '''Load datasets from <save_dir> files (3 modalities)'''
        train, eval, test  = self.combine_modalities(save_dir)

        # self.train_dataset = train.map(self.tokenize_function, remove_columns=["text"])\
        #     .filter(lambda example: len(example["input_ids"]) <= self.MAX_EMBEDDINGS)
        #     # .map(batched=True)

        # self.eval_dataset = eval.map(self.tokenize_function, remove_columns=["text"])\
        #     .filter(lambda example: len(example["input_ids"]) <= self.MAX_EMBEDDINGS)
        #     # .map(batched=True)

        self.test_dataset = test.map(self.tokenize_function, remove_columns=["text"])\
            .filter(lambda example: len(example["input_ids"]) <= self.MAX_EMBEDDINGS)
            #.map(batched=True)

        # self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        # self.eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        # self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples["text"], padding='max_length', truncation=False, return_tensors="pt")
        model_inputs["labels"] = squeeze(self.tokenizer(examples["labels"], padding='max_length', return_tensors="pt").input_ids) #TODO: do I tokenize this?
        model_inputs["attention_mask"] = squeeze(model_inputs["attention_mask"])
        model_inputs["input_ids"] = squeeze(model_inputs["input_ids"])

        return model_inputs


    def combine_modalities(self, save_dir):
        file_names  = ['data.parent_buggy_only', 'data.commit_msg', 'data.full_code_fullGraph', 'data.fixed_only']
        # file_names = ['data.buggy_only', 'data.commit_msg', 'data.full_code_leaveOnly', 'data.fixed_only']
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


    def train(self, pretrain_path: str = None):
        '''Finetune model using transformers Trainer class. Save final model in /models/model_name'''
        #initialize the transformers trainer
        config = {
            "output_dir": f"model/{self.model_name}.pt",
            "overwrite_output_dir": True,
            "evaluation_strategy": "steps",
            "generation_max_length": 512,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "save_strategy": 'epoch',
        }
        training_args = Seq2SeqTrainingArguments(**config)
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        if pretrain_path is not None:
            self.trainer.train(resume_from_checkpoint=pretrain_path)
        else: 
            self.trainer.train()


    def get_portions(self, batch_num: int, total: int):
        portions = total // batch_num
        res = []
        start = 0
        for _ in range(batch_num):
            res.append((start, start+portions))
            start += portions
                    
        return res


    def batch_train(self, batch_id: int, epoch: int, pretrained_model_path: str) -> str:
        #initialize the transformers trainer

        checkpoint_path = f"model/{self.model_name}_epoch_{epoch}_finetuned.pt"

        lr = pow(0.9992, batch_id+1) * pow(0.4, epoch+1) * 0.1
        print(f'Learning Rate: {lr}')

        if pretrained_model_path is not None:
            model = PLBartForConditionalGeneration.from_pretrained(pretrained_model_path, \
                                                                    local_files_only=True, \
                                                                    output_loading_info=False)
            # Cache Last time model
            self.model = model
        else:
            model = self.model


        config = { 
            "output_dir": f'./trainer/training_epoch{epoch}/batch{batch_id}',
            "evaluation_strategy": "epoch",
            "generation_max_length": 512,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "save_strategy": 'no',
            'num_train_epochs': 1,
            'learning_rate': lr,
            'logging_strategy': 'epoch',
            'log_level': 'info',
            'lr_scheduler_type': 'constant'
        }

        training_args = Seq2SeqTrainingArguments(**config)

    
        train_start, train_end = self.train_portions[batch_id]

        eval_start, eval_end = self.eval_portions[batch_id]
        
  
        # debug
        print("Training Portion: ",train_start, train_end)

        print("Eval Portion: ", eval_start, eval_end)
        
        train_dataset = torch.utils.data.Subset(self.train_dataset, list(range(train_start, train_end, 1)))
        eval_dataset = torch.utils.data.Subset(self.eval_dataset, list(range(eval_start, eval_end, 1)))
                
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        trainer.save_state()
        model.save_pretrained(checkpoint_path) 
        return checkpoint_path



    def evaluate(self):
        '''Evaluate model on test set by accuracy'''
        test_pred = self.trainer.predict(self.test_dataset)
        return self.compute_metrics(test_pred)


    def batch_evaluate(self, trainer):
        '''Evaluate model on test set by accuracy'''
        test_pred = trainer.predict(self.test_dataset)
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
            # if self.is_predction:
            print(preds[i] )
            # print(labels[i])
            print()
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

        return { 'acc': acc }


    def run(self, folder_name):
        '''Run finetuning'''
        print("Loading datasets...")
        self.load_datasets(folder_name)
        print("Finetuning...")
        self.train()
        print("Evaluating...")
        acc = model.evaluate()
        print("Accuracy: ", acc)

    
    def batch_run(self, folder_name: str, batch_num: int = 3):
        '''Run Batch finetuning'''
        print(f'''Dataset split uniformly by {batch_num} Portions''')
        print("Loading datasets...")
        self.load_datasets(folder_name)
        MAX_EPOCH = 10
        self.train_len = len(self.train_dataset)
        self.eval_len = len(self.eval_dataset)
        print("Training Sample:", self.train_len)
        print("Eval Sample:", self.eval_len)

        self.train_portions = self.get_portions(batch_num, self.train_len)
        self.eval_portions = self.get_portions(batch_num, self.eval_len)

        checkpoint_path =  None

        for epoch in range(MAX_EPOCH):
            batch_id = 0
            while batch_id != batch_num:
                print("Epoch", epoch)
                print("Finetuning", batch_id, "Portion")
                checkpoint_path = self.batch_train(batch_id, epoch, checkpoint_path)
                print("Cleaning Cache")
                cuda.empty_cache()
                batch_id += 1

                    
    
    def evaluate_pretrained_output(self, folder_name: str, batch_num: int = 3):     
        print("Evaluating...")
        self.load_datasets(folder_name)

        config = {
            "output_dir": "./trainer/training",
            "evaluation_strategy": "epoch",
            "generation_max_length": 512,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "save_strategy": 'no',
            'num_train_epochs': 1,
            'log_level': 'info',
        }
        
        training_args = Seq2SeqTrainingArguments(**config)


        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=self.compute_metrics,
        )
        
        batch_id = 0
        total_acc = 0
        print('Num of Dataset:', len(self.test_dataset))

        # pred = trainer.predict(self.test_dataset)
        # acc = self.compute_metrics(pred)['acc']
        portions = self.get_portions(batch_num, len(self.test_dataset))
        #        batch_num *= 2
        while batch_id != batch_num:
            test_start, test_end = portions[batch_id]
            print(test_start, test_end)
            test_dataset = torch.utils.data.Subset(self.test_dataset, list(range(test_start, test_end, 1)))
            test_pred = trainer.predict(test_dataset)
            total_acc += self.compute_metrics(test_pred)['acc']
            print(total_acc)
            batch_id += 1

        acc = total_acc / batch_num
        print("Accuracy: ", acc)        
        # print("Accuracy: ", total_acc)        


if __name__ == "__main__":
    model = Model("Modit-G-improved-Small", pretrained_model_path='model/Modit-G-improved_epoch_9_finetuned.pt')
    # model = Model("Modit-G-improved-Small", pretrained_model_path=None)

    # model = Model("Modit-G-improved-Medium", pretrained_model_path='lr-cp')
    # model.batch_run("./data/small", 750)
    #model.run("./data/small")
    # model.evaluate_pretrained_output('./data/medium', 600)
    model.evaluate_pretrained_output('./data/small', 500)