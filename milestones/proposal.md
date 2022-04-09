# Knowledge Representation for Code Patch
---
## Background
With large codebases, automatic code patching can increase the efficiency and decrease bugs for developers. We attempt to create a knowledge representation for such code patching either using a new graph structure, or combining existing ones such as an AST. Code patching can then be used downstream to find bugs, replace or improve code, or in this case, edit source code.
In MODIT, an encoder-decoder model was used to identify bugs given the code, natural language guidance, and context of the code. In giving natural language hints to the model, MODIT was able to improve on state of art models for code patching, and narrowed down the search space for such patches. In this paper we use a similar approach, and change the context representation of the code to increase the model’s understanding of code flow. We use the same preprocessed dataset as MODIT, a collection of bug-fix commits to GitHub. We will also first consider CodeT5, and determine which model is better suited towards our task.

## Methods
We will use similar inputs to MODIT, and consider alternative ways to represent the context.

- Feature 1: Code to be edited
- Feature 2: Guidance
- Feature 3: Context

In this paper we consider new ways to represent Modality 3, through simple sequences, AST, relational models, or some other graph that captures the flow and context of the code. We also consider Feature 4: code summarization. We will leverage knowledge of programming language by using a pretrained model, finetuned on input of the above features.


## Milestones
We split our timeline into 2 main goals. First, we want to understand existing models and investigate how well baseline models perform. We then will start our experiments to improve upon state-of-art models, by designing a relational graph representation of the code as the context.
### Milestone 1:
- Read existing literature
- Process dataset
- Run 2 models (MODIT, CodeT5)  
    - https://github.com/modit-team/MODIT
    - https://github.com/salesforce/CodeT5 

### Milestone 2:
- Design knowledge graph
- Experiment on best representations, and other hyperparameters to finetune a model.

## Relevant Papers
- Learn to represent program as graph https://arxiv.org/pdf/1711.00740.pdf 
- Global relational models of source code https://openreview.net/pdf?id=B1lnbRNtwr 
- PLUR: A Unifying, Graph-Based View of Program Learning, Understanding, and Repair https://openreview.net/pdf?id=GEm4o9A6Jfb 
- MODIT https://arxiv.org/pdf/2108.06645.pdf 
