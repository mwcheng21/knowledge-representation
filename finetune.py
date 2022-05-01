from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
import numpy as np
from itertools import zip_longest
from datasets import Dataset
from torch import squeeze, cuda
import torch
import os


class Model():
    def __init__(self, model_name, pretrained_tokenizer_path=None, pretrained_model_path=None):

        '''Config'''
        if(cuda.is_available()): print('GPU in using', cuda.current_device())

        '''Init function'''
        self.model_name = model_name
        #init pretrained model and tokenizer
        if pretrained_tokenizer_path==None:
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
            "save_strategy": 'no',
            "evaluation_strategy": "steps",
            "generation_max_length": 512,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1
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


    def get_portion(self, batch_id: int, total_batch_num: int, total_sample_size: int):
        portion_size = round(total_sample_size / total_batch_num)
        start = batch_id * portion_size
        end = start + portion_size
        return start, end


    def batch_train(self, batch_id: int, total_batch_num: int, pretrained_model_path: str, epoch: int) -> str:
        #initialize the transformers trainer
        config = {
            "output_dir": "./trainer/training",
            "evaluation_strategy": "steps",
            "generation_max_length": 512,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "save_strategy": 'no',
            'num_train_epochs': 1
        }

        # Update the model
        if pretrained_model_path is not None:
            assert os.path.exists(pretrained_model_path)
            model = PLBartForConditionalGeneration.from_pretrained(pretrained_model_path, \
                                                                    local_files_only=True, \
                                                                    output_loading_info=False)
        else:
            model = self.model

        train_start, train_end = self.get_portion(batch_id, total_batch_num, len(self.train_dataset))

        eval_start, eval_end = self.get_portion(batch_id, total_batch_num, len(self.eval_dataset))
  
        # debug
        print("Training Portion: ",train_start, train_end)

        print("Eval Portion: ", eval_start, eval_end)


        if batch_id < total_batch_num - 1:
            train_dataset = torch.utils.data.Subset(self.train_dataset, list(range(train_start, train_end, 1)))
            eval_dataset = torch.utils.data.Subset(self.eval_dataset, list(range(eval_start, eval_end, 1)))
        else:
            train_dataset = torch.utils.data.Subset(self.train_dataset, list(range(train_start, len(self.train_dataset), 1)))
            eval_dataset = torch.utils.data.Subset(self.eval_dataset, list(range(eval_start, len(self.eval_dataset), 1)))

        
        training_args = Seq2SeqTrainingArguments(**config)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()

        checkpoint_path = f"model/{self.model_name}_epoch_{epoch}_finetuned.pt"

        model.save_pretrained(checkpoint_path)  

        if batch_id == total_batch_num - 1:
            return checkpoint_path, trainer
        else:
            return checkpoint_path, None


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
            accur_count += 1 if preds[i] == labels[i]  else 0
        
        return accur_count / sample_size


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
        pretrain_path = None
        MAX_EPOCH = 5


        for epoch in range(MAX_EPOCH):
            batch_id = 0
            while batch_id != batch_num:
                print("Epoch", epoch)
                print("Finetuning", batch_id, "Portion")
                pretrain_path, trainer = self.batch_train(batch_id, batch_num, pretrain_path, epoch)
                print("Cleaning Cache")
                cuda.empty_cache()
                batch_id += 1


        print("Evaluating...")
        batch_id = 0
        total_acc = 0
        batch_num *= 2
        while batch_id != batch_num:
            test_start, test_end = self.get_portion(batch_id, batch_num, len(self.test_dataset))
            if batch_id < batch_num - 1:
                test_dataset = torch.utils.data.Subset(self.test_dataset, list(range(test_start, test_end, 1)))
            else:
                test_dataset = torch.utils.data.Subset(self.test_dataset, list(range(test_start,  len(self.test_dataset), 1)))
            test_pred = trainer.predict(test_dataset)
            total_acc += self.compute_metrics(test_pred)['Acc']
            batch_id += 1

        acc = total_acc / batch_num * 100
        print("Accuracy: ", acc)
    

    

if __name__ == "__main__":
    model = Model("Modit-G", pretrained_model_path="model/Modit-G_18_finetuned.pt")
    #model.batch_run("./data/medium", 1000)
    model.run("./data/medium")