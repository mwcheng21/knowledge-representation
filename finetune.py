from transformers import TrainingArguments
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import PLBartModel, PLBartConfig
from transformers import Trainer
import numpy as np
from datasets import load_metric

class Model():
    def __init__(self):
        #init pretrained model and tokenizer
        self.tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
        #TODO: what model to use PLBartModel or PLBartForConditionalGeneration?
        self.model = PLBartModel.from_pretrained("uclanlp/plbart-base")

        #use this metric in training
        self.metric = load_metric("accuracy")

    def preprocess(self):
        #TODO import dataset
        self.train_dataset = None
        self.eval_dataset = None
        #TODO: tokenize dataset
        #TODO: look at javalang or some other library to create AST??
        #TODO: append tokenized examples, summary, and context (hstack or something????)

    def train(self):
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

        #TODO: process or return output
        return None

    def evaluate(self):
        pass

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        #TODO: Not sure if this is right, since labels arent exactly the same, (what metric to use for accuracy???,)
        return self.metric.compute(predictions=predictions, references=labels)
