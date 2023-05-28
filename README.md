# Text-Summarizer-BART
This NLP project focuses on text summarization. It utilizes the BART pre-trained model on the CNN Daily Mail dataset as a base, and was further trained on the XSum dataset with Pytorch. It uses the rouge score as a metric to evaluate the produced summary against the references provided.

# Files
- BART Text Summarizer.py
- README.md
- Text Summarization Model on XSum - BART .py
- requirements.txt

# Download the Dataset
The XSum dataset summarizes over 220K+ BBC news articles and can be found at https://huggingface.co/datasets/xsum

# Project Steps
1. Import packages, dataset, and pre-trained model.
4. Preprocessing the data/ tokenization.
5. Creating the rouge score function.
6. Defining training arguments.
7. Training the model.
8. Evaluating and testing the model.

# How to use the model
'Text Summarization Model on XSum - BART .py' contains the code on loading, preprocessing, and training the model. For demonstration purposes, run 'BART Text Summarizer.py'. You may edit the text you would like to summarize there!

Feel free to also try out the model on https://huggingface.co/airinkonno/nlp_summarization_project.
