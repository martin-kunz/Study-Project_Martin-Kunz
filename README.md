# <ins>NER Robustness for Out of Domain Data</ins>

### <ins>Introduction</ins>
Named Entity Recognition (NER) is a vast and active field of research within Natural Language Processing (NLP) that deals with the identification and classification of key information in texts. This key information typically includes names of people, organizations, places, time and numerical values such as amounts of money and percentages. NER serves as an important building block for a variety of NLP applications, including information retrieval, question-answering systems, sentiment analysis, content summarization and automatic translation.<br>
However, there are often problems with the training and evaluation of NER models. Many models are trained and tested on the same data sets on which they were developed. As a result, the models deliver very good results within their domain, but often perform worse on out-of-distribution (OOD) data, meaning data outside their domain, for example, if a NER model is trained on the CoNLL-2003 dataset [[1](https://aclanthology.org/W03-0419/)] and also evaluated on test data from this data, it delivers very good results. However, if we now use test data from the SEC dataset and evaluate it on a model trained on CoNLL-2003, we observe a significant drop in performance compared to testing on CoNLL-2003's test data. While the CoNLL-2003 test data yields a precision of 0.833, recall of 0.824, and F1 score of 0.829, testing on SEC results in a drastic reduction to precision of 0.247, recall of 0.132, and F1 score of 0.172 [[2](https://aclanthology.org/U15-1010/)].<br>
The goal of this research is to modify the last linear layer of our model, which is responsible for the final prediction of the individual entities, using different initialization techniques. Different modification techniques will be applied and evaluated in order to generate the best possible results for OOD datasets.

### <ins>Approach</ins>
This project aims to investigate Named Entity Recognition (NER) using a modified XLM-RoBERTa model. In essence, the prediction head of the pre-trained model was replaced to initialize the classification layer with new, most frequent token embeddings. The modified model was then evaluated to investigate how this adaptation affects performance compared to an unmodified baseline model (training and evaluation on CoNLL-2003 dataset).<br>
The key steps include processing the CoNLL-2003 dataset, fine-tuning the model on this data and then analyzing the predictions after modifying the prediction head. In addition to conducting the baseline experiment, specific methods were implemented to identify the most generic tokens for each entity class and integrate them into the new classification layer.<br>
The individual scripts are then explained in detail, including data preprocessing, model initialization, modification of the prediction head and evaluation. These explanations are intended to make the structure and workflow of the project comprehensible.

### <ins>Scripts and Data</ins>
- ```masking.py``` - Replaces tokens in a data set that are tagged with certain NER tags with a placeholder (`[MASK]`). This script creates a masked output for each tag for the test, train and validation datasets.
- ```data_preprocessing.py``` - Contains functions for preparing data for NER tasks. The script tokenizes texts, matches NER tags to tokens and provides methods to replace specific tags with masks. Predictions for masked tokens can also be extracted.
- ```topk_predictions.py``` - Calculates the top-k predictions for masked tokens in a text. The script uses the XLM-RoBERTa model to generate predictions for masked sentences in the training, validation and test datasets.
- ```combine.py``` - This script combines prediction data from multiple tag-specific text files into a single TSV file. It creates a structured output with columns for different tags such as ```I-PER```, ```B-LOC``` etc. and is designed for the train, test and validation data sets.
- ```get_top_predictions.py``` - Extracts the most frequent predictions for NER tags from several files and creates a cleaned output containing the best predictions for each tag.
- ```top_ner_predictions.py``` - Counts the most frequent first predictions for masked tokens from a CSV file and saves the results as a text file. The output contains the prediction, its frequency and the percentage.
- ```topk_predictions.py``` - Calculates the top-k predictions for masked tokens in a text. The script uses the XLM-RoBERTa model to generate predictions for masked sentences in the training, validation and test datasets.
- ```train_xlmroberta.py``` - Trains an XLM-RoBERTa model for NER tasks with the CoNLL03 dataset. It defines training arguments, processes the data and saves the trained model as well as checkpoints and logs.
- ```replace_prediction_head.py``` - Replaces the classification head of an XLM-RoBERTa model so that it provides static predictions for each tag. This is used to analyze the model parameters and the influence of specific initializations.
- ```eval.py``` - Evaluates a trained NER model using the CoNLL03 dataset. It calculates metrics such as Precision, Recall, F1-Score and Accuracy. The script uses the `Trainer` of HuggingFace for the evaluation and saves the results in a log file.
- ```/data``` - Contains the data sets used to train the model.
- ```/masking_results``` - Contains the results for the top predictions of each of the nine tags of the train, valid and test dataset.