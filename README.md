# LePaRD: A Large-Scale Dataset of Judges Citing Precedents

This repo contains the code and models for the paper LePaRD paper [(Mahari et al., 2024)](https://aclanthology.org/2024.acl-long.532/).

## Updated dataset and results

Updated dataset statistics and results (the output of the repliation package in this repo) can be found in the arxiv version [(Mahari et al., 2024)](https://arxiv.org/abs/2311.09356). This updated paper and code includes improvements that have been implemented since LePaRD was published in ACL.

Download the dataset on [Hugging Face](TBD)

# Reference

Please cite the following paper if you use LePaRD:

```bibtex
@inproceedings{mahari-etal-2024-lepard,
    title = "{L}e{P}a{RD}: A Large-Scale Dataset of Judicial Citations to Precedent",
    author = "Mahari, Robert  and Stammbach, Dominik  and Ash, Elliott  and Pentland, Alex",
    editor = "Ku, Lun-Wei  and Martins, Andre  and Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.532",
    pages = "9863--9877",
}
```

# Description

LePaRD is a massive collection of U.S. federal judicial citations to precedent in context. LePaRD builds on millions of expert decisions by extracting quotations to precedents from judicial opinions along with the preceding context. Each row of the dataset corresponds to a quotation to prior case law used in a certain context.

- passage_id: A unique idenifier for each passage
- destination_context: The preceding context before the quotation
- passage_text: The text of the passage that was quoted
- court: The court from which the passage originated
- date: The date when the opinion from which the passage originated was published

Contact [Robert Mahari](https://robertmahari.com/) in case of any questions.


## Data
The original data can be downloaded here: https://www.dropbox.com/scl/fo/0pgqxcz0h2l4wta8yyvb3/ABjE8bNAnq3Vm2bBJziclPE?rlkey=zipkfcso0h9je1xne737ims02&st=6mgtpwa0&dl=0
To run the replication package, make sure to store all files in a folder called *data*.

## Installation

Requires
* For bm25: [anserini](https://github.com/castorini/anserini)
* For dense retrieval: [SBERT](https://github.com/UKPLab/sentence-transformers/) and [Faiss](https://github.com/facebookresearch/faiss)
* For classification experiments: [transformers](https://huggingface.co/docs/transformers/installation)

## Experiments

First, split the data into train, dev and test. The output of this process can also be downloaded [here](https://www.dropbox.com/scl/fi/m4z379fyjgi33ppu8q0fs/data_postprocessed.zip?rlkey=nrhton2dkku9gdv8alcj0g7f1&st=mpay7kqc&dl=0).

```shell
python src/prepare_data.py
```

### bm25 experiments

```
# reformat input files
python src/bm25_pipeline.py

# run anserini and bm25 retrieval
path_anserini="/path/to/anserini"
num_labels="10000" # change this to 20000 / 50000 for other experiments

# build index
sh $path_anserini/target/appassembler/bin/IndexCollection -threads 1 -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator -input bm25-files-$num_labels \
 -index indexes/index-lepard-passages-$num_labels -storePositions -storeDocvectors -storeRaw 

# retrieve passages devset
sh $path_anserini/target/appassembler/bin/SearchMsmarco -hits 10 -threads 1 \
 -index indexes/index-lepard-passages-$num_labels \
 -queries bm25-files-$num_labels/bm25_input_dev_$num_labels".tsv" \
 -output bm25-files-$num_labels/bm25_output_dev.tsv/

# retrieve passages testset
sh $path_anserini/target/appassembler/bin/SearchMsmarco -hits 10 -threads 1 \
 -index indexes/index-lepard-passages-$num_labels \
 -queries bm25-files-$num_labels/bm25_input_test_$num_labels".tsv" \
 -output bm25-files-$num_labels/bm25_output_test.tsv/

# evaluate
python src/evaluate_run.py --dev_predictions bm25-files-$num_labels/bm25_output_dev.tsv --test_predictions bm25-files-$num_labels/bm25_output_test.tsv --experiment bm25
```

### classification experiments

```
num_labels="10000" # change this to 20000 / 50000 for other experiments
model_name="distilbert-base-uncased"
python src/train_classification_models.py --n_labels 10000 --model_name $model_name # trains default distilbert models and saves model and predictions in folder "finetuned-$model_name-$num_labels"
# evaluate
python src/evaluate_run.py --dev_predictions finetuned-distilbert-base-uncased-10000/predictions_devset_10000.json --test_predictions finetuned-distilbert-base-uncased-10000/predictions_testset_10000.json 
# for all experiments, change num_labels to 20000 and 20000, and also run with legalbert 
```

### SBERT experiments
```
# zero-shot
num_labels="10000" # change this to 20000 / 50000 for other experiments
model_name="sentence-transformers/all-mpnet-base-v2"
python src/run_inference_sbert.py --model_name $model_name n_labels $num_labels # creates folder "predictions-sbert" and saves output there (os.path.basename(model_name) + predictions dev/test)
# evaluate
python src/evaluate_run.py --dev_predictions predictions-sbert/predictions_devset_all-mpnet-base-v2_$num_labels.json --test_predictions predictions-sbert/predictions_testset_all-mpnet-base-v2_$num_labels.json
```

```
# fine-tune
num_labels="10000" # change this to 20000 / 50000 for other experiments
python src/finetune_sbert.py --n_labels $num_labels # saves model in "sbert-finetuned-MultipleNegativesRankingLoss" + os.path.basename(args.model_name) + "-" + args.n_labels
$model_name="finetuned-distilbert-base-uncased-10000"
# run inference
python src/run_inference_sbert.py --model_name $model_name n_labels $num_labels # creates folder "predictions-sbert" and saves output there (os.path.basename(model_name) + predictions dev/test)
# evaluate
python src/evaluate_run.py --dev_predictions finetuned-distilbert-base-uncased-10000/predictions_devset_10000.json --test_predictions finetuned-distilbert-base-uncased-10000/predictions_testset_10000.json 
```

### Data replication
The training data may be generated using the Case Law Access project dataset using the precedent_data_extraction.py script
