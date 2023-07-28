import argparse
import transformers
from datasets import load_dataset


model_checkpoint = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": ["### End", "### Instruction:", "### Response:\n"]
})

def parse_args():
    """
    This function parses the arguments to the script from the jupyter notebook cell:
    
    Inputs:
        None
    Outputs:
        parser.parse_args()
    """
    # Create an ArgumentParser Object
    parser = argparse.ArgumentParser(description = "Process")
    
    parser.add_argument("--input-max-length", type = int, default = 1024)
    parser.add_argument("--output-max-length", type = int, default = 64)
    parser.add_argument("--output-data", type = str, default = "/opt/ml/processing/output")
    
    return parser.parse_args()


def tokenize(examples):
    inputs = ["summarize: " + article for article in examples["document"]]
    model_inputs = tokenizer(inputs, 
                             max_length = input_max_length, 
                             truncation = True)
    
    labels = tokenizer(text_target = examples["summary"],
                       max_length = output_max_length,
                       truncation = True)
    
    model_inpts["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    args = parse_args()
    
    data = load_dataset("xsum")
    
    data = data.filter(lambda example, index: index % 100 == 0, with_indices = True)
    
    tokenized_data = data.map(tokenize, batched = True, remove_columns = )