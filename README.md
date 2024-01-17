## Product Features Extraction

This project aims to build a framework, leveraging seq2seq models, for extracting attribute values. 
The framework is designed to handle various attribute types, including Boolean features (true/false) and general open-domain attributes such as dimension, width, color, and more.

#### Requirements

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   Ensure that the data for experiments follows the expected format. For details, look into [data_loader](src/data_processing/data_loader.py) 

Or use this command to create a new conda environment with all dependencies:

 ```bash
conda env create -f environment.yml
 ```
#### To reproduce experiments:
1. Run the training script using the provided run_experiment.sh file. Make sure to specify the data path as an environment variable.
2. In the [run_experiment.sh](run_experiment.sh) file, set the desired parameters for the script. Use boolean parameters by calling them to set them to true, and omit them to set them to false.
  Parameters are: 
   - path: path to data
   - batch_size
   - multi_task: true for a multi-task setting, false for a single task
   - bf: true to train the model uniquely on binary attributes (requires multi-task=false)
   - da: set to true to add augmented examples
   - insert_ng: set to true to insert negated samples
   - clust: true to use clustering as a coreset selection method
   - device: cuda device
   - epochs: number of training epochs, default=3
3. Execute the script from the terminal
    ```bash
   sh run_experiment.sh
Alternatively, you can define environment variables and specify parameters directly in the terminal, following the structure outlined in the run_experiment.sh file.

#### Miscellaneous
- For cross-validation: [kfold](src/kfold.py)
- To test models: [test](src/test.py)
- The Data Augmentation script: [DA](src/data_processing/data_augmentation.py)
- Main hyperparameters and settings are found in: [config](src/config.json)
- To use prefix-tuning as a peft method use the configuration in: [prefix_tuning](src/peft_prefix_tuning.py)

#### References

- https://huggingface.co/
- https://spacy.io/
- https://pytorch.org/
