from transformers import TrainingArguments
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import PLBartModel, PLBartConfig
from transformers import Trainer
import numpy as np
from datasets import load_metric

class Model():
    def __init__(self, pretrained_tokenizer_path=None, pretrained_model_path=None):
        '''Init function'''
        #init pretrained model and tokenizer
        if pretrained_model_path==None:
            self.tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
        else:
            #TODO: make sure this works
            self.tokenizer = PLBartTokenizer.from_pretrained(pretrained_tokenizer_path, local_files_only=True)


        #TODO: what model to use PLBartModel or PLBartForConditionalGeneration?
        if pretrained_model_path==None:
            self.model = PLBartModel.from_pretrained("uclanlp/plbart-base")
        else:
            #TODO: load model
            pass

        #use this metric in training
        self.metric = load_metric("accuracy")

    def preprocess(self, save_dir):
        '''Loads src code, preprocesses and saves into files in save_dir'''
        #TODO import dataset
        self.train_dataset = None
        self.eval_dataset = None
        #TODO: look at javalang or some other library to create AST??
        #TODO: save AST into files so we don't have to preprocess later

    def load_datasets(self, save_dir):
        '''Combines 3 modalities into vectors'''
        pass
        #TODO: append tokenized examples, summary, and context (hstack or something????)

    def train(self, save_file):
        '''Finetune model using transformers Trainer class. Save final model in <save_file>'''
        #initialize the transformers trainer
        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        #train it #TODO: we can also write fine tuning by hand using pytorch
        trainer.train()

        #TODO: make sure to save the model

        #TODO: process or return output
        return None

    def evaluate(self):
        '''Evaluate eval set on self.model'''
        pass

    def compute_metrics(self, eval_pred):
        '''Helper function for what metric training should use'''
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        #TODO: Not sure if this is right, since labels arent exactly the same, (what metric to use for accuracy???,)
        return self.metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    model = Model()
    #TODO: uncomment as needed to test
    # model.preprocess()
    # model.load_datasets()
    # model.train()
    # model.evaluate()