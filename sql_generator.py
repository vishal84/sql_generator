import json
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.embeddings.base import Embeddings
from langchain.chat_models import ChatVertexAI
from langchain.vectorstores import FAISS
from google.cloud import bigquery
from typing import List
from tqdm import tqdm


PROJECT = 'gcp-sre-edu-image-host'
LOCATION = 'us-central1'
CHAT_MODEL = 'codechat-bison@002'
EMBEDDING_MODEL = 'textembedding-gecko@002'
TABLES_LOC = './data/tables.jsonl'
COLS_LOC = './data/columns.jsonl'
DEFNS_LOC = './data/defns.jsonl'

class VertexAIEmbeddingsModel(VertexAIEmbeddings, Embeddings):
    max_batch_size = 5
    
    def embed_segments(self, segments: List) -> List:
        embeddings = []
        for i in tqdm(range(0, len(segments), self.max_batch_size)):
            batch = segments[i: i+self.max_batch_size]
            embeddings.extend(self.client.get_embeddings(batch))
        return [embedding.values for embedding in embeddings]
    
    def embed_query(self, query: str) -> List:
        embeddings = self.client.get_embeddings([query])
        return embeddings[0].values

def _create_vector_store(file_loc, embedding, search_type, search_kwargs):
    
    documents = JSONLoader(file_path=file_loc, jq_schema='.', 
                           text_content=False, json_lines=True).load()
    db = FAISS.from_documents(documents=documents, embedding=embedding)
    retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    return db, retriever

def setup_dbs(embedding, file_loc, score_threshold=0.25):
    
    db, retriever = _create_vector_store(file_loc, embedding,
                                         'similarity_score_threshold', 
                                         {'score_threshold': score_threshold})
    
    return db, retriever

def create_chat_model(project, location, chat_model, temp=0.0, max_output_tokens=512):
    
    llm = ChatVertexAI(project=project, 
                       location=location, 
                       model_name=chat_model,
                       temperature=0.0, 
                       max_output_tokens=512)
    
    return llm

def match_tables(query, retriever):
    
    matched_documents = retriever.get_relevant_documents(query=query)

    matched_tables = []

    for document in matched_documents:
        page_content = document.page_content
        page_content = json.loads(page_content)
        table_name = page_content['table_name']
        matched_tables.append(f'{table_name}')
    
    return matched_tables
    
def match_columns(query, retriever, matched_tables):
    
    matched_columns = retriever.get_relevant_documents(query=query)
    
    matched_columns_filtered = []

    # LangChain filters does not support multiple values at the moment
    for tbl_name in matched_tables:
        for i, column in enumerate(matched_columns):
            page_content = json.loads(column.page_content)
            table_name = page_content['table_name']
            if table_name == tbl_name:
                matched_columns_filtered.append(page_content)
    
    matched_columns_cleaned = []

    for doc in matched_columns_filtered:
        dataset_name = doc['dataset_name']
        table_name = doc['table_name']
        column_name = doc['column_name']
        data_type = doc['data_type']
        description = doc['Description']
        matched_columns_cleaned.append(f'dataset_name={dataset_name}|table_name={table_name}|column_name={column_name}|data_type={data_type}|description={description}')

    matched_columns_cleaned = '\n'.join(matched_columns_cleaned)
    
    return matched_columns_cleaned

def match_defns(query, retriever):
    
    matched_documents = retriever.get_relevant_documents(query=query)

    matched_defns = []

    for document in matched_documents:
        page_content = document.page_content
        page_content = json.loads(page_content)
        term = page_content['term']
        definition = page_content['definition']
        abbreviations = page_content['abbreviations']
        matched_defns.append(f'term={term}|definition={definition}|abbreviations={abbreviations}')
    
    return matched_defns

def build_chat_request(query, matched_columns, matched_defns):
    
    messages = []
    template = "You are a SQL master expert capable of writing complex SQL queries in BigQuery."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    messages.append(system_message_prompt)

    human_template = """Given the following inputs:
    USER_QUERY:
    --
    {query}
    --
    MATCHED_SCHEMA: 
    --
    {matched_schema}
    --
    HELPFUL_DEFINITIONS: 
    --
    {matched_defns}
    --
    Please construct a SQL query using ONLY the MATCHED_SCHEMA and the USER_QUERY provided above. If a column is not listed in MATCHED_SCHEMA, then it should not be used! The user_id column should always be used when counting users, learners or trained individuals. Please comment your SQL to explain the code you are writing and DO NOT USE STRING FUNCTIONS ON DATE OR TIMESTAMP FIELDS.
    
    The ultimate goal is to answer questions about learner data for Google Cloud's Cloud Learning Services.  

    IMPORTANT: Use ONLY the column names (column_name) mentioned in MATCHED_SCHEMA. Do not use ANY columns which are not provided in MATCHED_SCHEMA
    IMPORTANT: Associate column_name mentioned in MATCHED_SCHEMA only to the table_name specified under MATCHED_SCHEMA.
    IMPORTANT: Please ensure the description makes sense for the columns being used.
    IMPORTANT: All table references must be of the form concord-prod.dataset_name.table_name inside the FROM clause.
    IMPORTANT: UNION is not a valid operator. You must use UNION ALL or UNION DISTINCT
    IMPORTANT: Use SQL 'AS' statement to assign a single word new name temporarily to a table column or even a table wherever needed.  
    """
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    messages.append(human_message)
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    
    request = chat_prompt.format_prompt(query=query,
                                        matched_schema=matched_columns,
                                        matched_defns=matched_defns).to_messages()
    
    return request

def build_sql_query(chat_request, llm):
    response = llm(chat_request)
    sql = '\n'.join(response.content.strip().split('\n')[1:-1])
    
    return sql
    
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    
    print("\n\nInitializing models and vector stores.\n\n")
    
    llm = create_chat_model(PROJECT, LOCATION, CHAT_MODEL, temp=0.0, max_output_tokens=512)
    embedding = VertexAIEmbeddingsModel()
    
    db_tables, retriever_tables = setup_dbs(embedding, TABLES_LOC)
    db_cols, retriever_cols = setup_dbs(embedding, COLS_LOC)
    db_defns, retriever_defns = setup_dbs(embedding, DEFNS_LOC)

    while True:
        
        query = input("Input your question: ")

        print("\n\nIdentifying tables and columns for query. \n\n")
        matched_tables = match_tables(query, retriever_tables)
        matched_columns = match_columns(query, retriever_cols, matched_tables)
        matched_defns = match_defns(query, retriever_defns)
        
        if args.debug:
            print('Debug Information:\n  Tables:')
            for table in matched_tables:
                print(table)
            print(f'\n\n  Matched Columns = {matched_columns}\n\n  Definitions:')
            for defn in matched_defns:
                print(defn)

        chat_request = build_chat_request(query, matched_columns, matched_defns)

        print(f"Building SQL Query for the question: {query} \n\n")
        sql = build_sql_query(chat_request, llm)

        print(sql + '\n\n')
        
        x = input("Type 'yes' if you wish to continue: ")
    
        
        if x == 'yes' or x == 'Yes':
            continue
        else:
            print("\n\nGoodbye.\n\n")
            break
