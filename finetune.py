from transformers import TrainingArguments
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import PLBartModel, PLBartConfig
from transformers import Trainer
import numpy as np
from datasets import load_metric
from nltk.translate.bleu_score import corpus_bleu


class Model():
    def __init__(self, model_name, pretrained_tokenizer_path=None, pretrained_model_path=None):
        '''Init function'''
        self.model_name = model_name
        #init pretrained model and tokenizer
        if pretrained_model_path==None:
            self.tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
        else:
            self.tokenizer = PLBartTokenizer.from_pretrained(pretrained_tokenizer_path, local_files_only=True)

        #TODO: what model to use PLBartModel or PLBartForConditionalGeneration? Not sure the difference
        if pretrained_model_path==None:
            self.model = PLBartModel.from_pretrained("uclanlp/plbart-base")
        else:
            self.model = PLBartModel.from_pretrained(pretrained_model_path, local_files_only=True)

        self.metric = load_metric("bleu")

    def load_datasets(self, save_dir):
        '''Load datasets from <save_dir> files (3 modalities)'''

        #Load into a DataLoader
        self.train_dataset = None
        self.eval_dataset = None
        pass
        #TODO: append tokenized examples, summary, and context (hstack or something????)


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

    #TODO: do we evaluate bleu_1, 2, 3 and 4? and rouge_L and CIDEr?
    def evaluate(self):
        '''Evaluate model on test set'''
        preds = self.model.predict(self.eval_dataset)
        bleuScore = corpus_bleu(self.eval_dataset.targets, preds)
        return bleuScore

    def compute_metrics(self, eval_pred):
        '''Helper function for what metric training should use'''
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def run(self):
        '''Run finetuning'''
        model.load_datasets("basic_ast_small")
        model.train()
        bleuScore = model.evaluate()
        print("Bleu Score: ", bleuScore)

if __name__ == "__main__":
    model = Model()
    model.run()