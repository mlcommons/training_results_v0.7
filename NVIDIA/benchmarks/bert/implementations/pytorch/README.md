# Location of the input files 

This [GCS location](https://console.cloud.google.com/storage/browser/pkanwar-bert) contains the following.
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

# Download and preprocess datasets

Download the [wikipedia dump](https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2) and extract the pages
The wikipedia dump can be downloaded from this link in this directory, and should contain the following file:
enwiki-20200101-pages-articles-multistream.xml.bz2

Run [WikiExtractor.py](https://github.com/attardi/wikiextractor) to extract the wiki pages from the XML
The generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has 100 files from wiki_00 to wiki_99, except the last sub directory. For the 20200101 dump, the last file is FE/wiki_17.

Clean up
The clean up scripts (some references here) are in the scripts directory.
The following command will run the clean up steps, and put the results in ./results
./process_wiki.sh '<data dir>/*/wiki_??'

```shell
cd cleanup_scripts  
mkdir -p wiki  
cd wiki  
wget https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2    # Optionally use curl instead  
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2  
cd ..    # back to bert/cleanup_scripts  
git clone https://github.com/attardi/wikiextractor.git  
python3 wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml    # Results are placed in bert/cleanup_scripts/text  
./process_wiki.sh '<text/*/wiki_??'  
```

After running the process_wiki.sh script, for the 20200101 wiki dump, there will be 500 files, named part-00xxx-of-00500 in the ./results directory.

# Checkpoint conversion
python convert_tf_checkpoint.py --tf_checkpoint /cks/model.ckpt-28252 --bert_config_path /cks/bert_config.json --output_checkpoint model.ckpt-28252.pt

# Generate the BERT input dataset

The create_pretraining_data.py script duplicates the input plain text, replaces different sets of words with masks for each duplication, and serializes the output into the TFRecord file format. 

```shell
python3 create_pretraining_data.py \
   --input_file=<path to ./results of previous step>/part-XX-of-00500 \
   --output_file=<tfrecord dir>/part-XX-of-00500 \
   --vocab_file=<path to vocab.txt> \
   --do_lower_case=True \
   --max_seq_length=512 \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
```

The generated tfrecord has 500 parts, totalling to ~365GB.

# Running the model

Building the Docker container
```shell
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

To run this model, use the following command. Replace the configuration script to match the system being used (e.g., config_DGX1.sh, config_DGX2.sh, config_DGXA100.sh).

```shell
source config_DGXA100.sh
sbatch -N${DGXNNODES} --ntasks-per-node=${DGXNGPU} --time=${WALLTIME} run.sub
```

Alternative launch with nvidia-docker. Replace the configuration script to match the system being used (e.g., config_DGX1.sh, config_DGX2.sh, config_DGXA100.sh).

```bash
docker build --pull -t mlperf-nvidia:language_model .
source config_DGXA100.sh
CONT=mlperf-nvidia:language_model DATADIR=<path/to/datadir> DATADIR_PHASE2=<path/to/datadir_phase2> EVALDIR=<path/to/evaldir> CHECKPOINTDIR=<path/to/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1> ./run_with_docker.sh
```

For multi-node training, we use Slurm for scheduling and Pyxis to run our container.

## Configuration File Naming Convention

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>x<GRADIENT_ACCUMULATION_STEPS>.sh`.

### Example 1
A DGX1 system with 1 node, 8 GPUs per node, batch size of 6 per GPU, and 6 gradient accumulation steps would use `config_DGX1_1x8x6x6.sh`.

### Example 2
A DGX A100 system with 32 nodes, 8 GPUs per node, batch size of 20 per GPU, and no gradient accumulation would use `config_DGXA100_32x8x20x1.sh`

# Acknowledgements

We'd like to thank members of the ONNX Runtime team at Microsoft for their suggested performance optimization to reduce the size of the last linear layer to only output the fraction of tokens that participate in the MLM loss calculation.



