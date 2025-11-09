# IR_llm_negation_rewriting
Project for our project in the Information Retrieval course at Radboud University.


# LLM Rewriting:
The LLM rewriting is done using the llm_extractinator framework available at https://github.com/DIAGNijmegen/llm_extractinator.
To work with it, setup the conda environment as shown in their instructions.
Then, once you have copied this repository, simply create 3 folders in the llm_extractinator directory, named "NevIR", "QUEST", and "ExcluIR".
In each of these directories, ensure you have a "data" folder, with the dataset files named as in the top of the `process_datasetname_part1.py` files.
Then, simply run each part 1 file and run the llm_extractinator as instructed via the displayed output. Afterwards, run the `part2.py` files to collect and aggregate the results.
