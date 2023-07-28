# T5-News-Summarizer

Fine-tuning and deploying a T5 LLM on Amazon SageMaker to perform summarization on News articles.

## Project Motivation

The motivation for this project was to gain experience fine-tuning LLMs.

## Repository Structure and File Description

```markdown
├── notebooks                                     # Folder containing the jupyter notebooks
│   ├── Preprocessing.ipynb                       # Notebook for preprocessing the data
│   ├── fine_tuning.ipynb                         # Notebook for training and fine-tuning LLM model
│   ├── deployment.ipynb                          # Notebook for deploying fine-tuned LLM model
├── src     
│   ├── process_data.py                           # processing data script
│   ├── process_data.py                           # training model script
├── .gitignore                                    
├── LICENSE                                    
├── README.md                                     # Readme file
├── requirements.txt                              # requirements file           

```

## Requirements

You will need an AWS account to run the following notebooks.

## How To Interact With the Project

1. Clone the repository by running the following command:
```
https://github.com/zhikaichen99/T5-News-Summarizer.git
```
2. Run the notebooks in SageMaker