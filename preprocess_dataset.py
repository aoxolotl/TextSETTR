from datasets import load_dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import spacy
import timeit

from transformers import HfArgumentParser


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    
    train_file_path: Optional[str] = field(
        default= None,
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default= None,
        metadata={"help": "Path for cached valid dataset"},
    )
    
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )

if __name__ == '__main__':
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    raw_datasets = DatasetDict()

    # val split
    raw_datasets["validation"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=f"train[:{data_args.validation_split_percentage}%]",
        cache_dir=data_args.cache_dir,
    )

    # train split
    raw_datasets["train"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=f"train[{data_args.validation_split_percentage}%:]",
        cache_dir=data_args.cache_dir,
    )

    def split_reviews(dataset):
        """
        Split reviews into example sentences
        """
        reviews = dataset['review_body']
        
        docs = nlp.pipe(reviews, disable=['tok2vec', 'tagger', 'parser', 'lemmatizer'])

        sents = [list(doc.sents)[:2] for doc in docs]

        # Filter running slow 
        #sents = list(filter(lambda x: len(x) >= 2, sents))
        sents = [sent for sent in sents if len(sent) == 2]

        outdict = {'inputs':[sent[0].text for sent in sents], 
                   'contexts':[sent[1].text for sent in sents]}

        return outdict


    # Spacy sentencizer (better than a naive regex split)
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')

    # Pick column names from train for remove_columns
    columns = raw_datasets['train'].column_names
    dataset = raw_datasets.map(
        split_reviews,
        batched=True,
        remove_columns=columns,
        desc="Running sentencizer on review_body of dataset",
    )

    if data_args.cache_dir is None:
        dataset.save_to_disk(f'{data_args.dataset_name}')
    else:
        dataset.save_to_disk(f'{data_args.cache_dir}')
