import argparse
import logging
import os
import numpy as np

import evaluate
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

# using rouge metric to compute similarity between generated summaries and actual summaries
rouge = evaluate.load("rouge")


def parse_args():
    """
    This function parses the arguments to the script from the jupyter notebook cell
    
    Inputs:
        None
    Ouputs:
        parser.parse_args()
    """
    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--epochs", type = int, default = 10)
    parser.add_argument("--learning-rate", type = float, default = 1e-6)
    parser.add_argument("--train-batch-size", type = int, default = 8)
    parser.add_argument("--eval-batch-size", type = int, default = 8)
    parser.add_argument("--model-name", type = str)
    parser.add_argument("--evaluation-strategy", type=str, default="epoch")
    parser.add_argument("--save-strategy", type=str, default="no")
    parser.add_argument("--save-steps", type=int, default=500)

    
    # directories
    parser.add_argument("--output-data-dir", type = str, default = os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type = str, default = os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type = str, default = os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid-dir", type = str, default = os.environ["SM_CHANNEL_VALID"])
    
    
    
    return parser.parse_args()

def compute_rouge_score(eval_pred):
    """
    Compute ROUGE scores
    
    Inputs:
        generated (str) - generated summaries from the LLM
        reference (str) - reference summaries from the data
    """
    generated, reference = eval_pred
    
    decoded_generated = tokenizer.batch_decode(generated, skip_special_tokens = True)
    reference = np.where(reference != -100, reference, tokenizer.pad_token_id)
    decoded_reference = tokenizer.batch_decode(reference, skip_special_tokens = True)
    result = rouge.compute(predictions = decoded_generated, references = decoded_reference, use_stemmer = True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in generated]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

    


if __name__ == "__main__":
    args = parse_args()
    
    train_data = load_from_disk(args.train_dir)
    validation_data = load_from_disk(args.valid_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Training Set: {}".format(train_data))
    logger.info("Validation Set: {}".format(validation_data))
    
    # load model and tokenizer from hugging face
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)
    
    # define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir = args.model_dir,
        num_train_epochs = args.epochs,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.eval_batch_size,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        predict_with_generate=True,
        fp16=True,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        tokenizer = tokenizer,
        train_dataset = train_data,
        eval_dataset = validation_data,
        data_collator = data_collator,
        compute_metrics = compute_rouge_score,
    )
    
    # train model
    trainer.train()
    
    # save model to s3
    trainer.save_model(args.model_dir)