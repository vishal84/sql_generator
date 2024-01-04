# sql_generator
A Generative AI SQL code generator for creating Cloud Learning Services queries.

This tool is used to leverage Generative AI to write SQL code which can be utilized in Google BigQuery for querying the Cloud Skills Boost platform's datawarehouse.

## Usage

To run the program use any shell with `gcloud` configured as an authorized user for the `PROJECT` variable referenced in the Python script of this repo. Setup a `virtualenv` with at least Python 3.10 and install dependencies found in the `requirements.txt` file. Thereafter, run the program using `python sql_generator.py`. The script will initialize and prompt you for a human readable query which will be input to the LLM `codechat-bison@002`. 

