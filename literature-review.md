# Literature Review

## Learning to Represent Programs with Graphs
https://arxiv.org/pdf/1711.00740.pdf

### Overview 
**model architecture:** GG-NN ([code](https://github.com/Microsoft/gated-graph-neural-network-samples))
**data:** 2.9m LoC, 29 big projects
**evaluation on tasks:**
- new task proposed: VarMissuse, benchmarked agains BiRNN baseline
- old task used VarName, benchmarked against previous models
**results:**
- 32.9% accuracy on the VarNaming
- 85.5% accuracy on the VarMisuse task
- several real bugs in OSS projects were fixed
**constraints:** statically typed languages, C#

### Summary
The important part for us is the graph representation (see sec 4. from the paper). They use a "program graph" with syntax, data-flow, and type information (see below from the paper).
![Program Representation](/img/ggnn.png "Program Representation")

They get the syntatic information from AST and combine it sematic information using the data and control flow. They also use 10 types of edges listed below:
Child/NextToken
- LastRead/LastWrite/ComputedFrom - variable edges, models control/data flows
- LastLexicalUse - shows repeated use of same variable
- GuardedBy/GuardedByNegation - enclosing guard expression for this variable
- FormalArgName — connects method call arguments to the name/type declaration
- ReturnsTo — links return tokens to name/type in method declaration

### Architecture
They use Gated Graph Neural Network from [here](https://arxiv.org/abs/1511.05493) but for us this isn't as important.

Input: graph
Output: sequence

## Global Relational Models of Source Code
https://openreview.net/pdf?id=B1lnbRNtwr

### Overview 
wip

### Summary
wip





















