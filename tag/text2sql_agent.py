import argparse
import json
import os
import re
import sqlite3
import time
import traceback
import pandas as pd
from collections import defaultdict
import numpy as np
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tag.utils import row_to_str, eval

TEXT2SQL_PROMPT = """
Given the following SQL schema and the provided external knowledge, write a SQL query to answer the question.
- The SQL should start with `SELECT`.
- Output only the SQL query without additional explanation.
-----
### DB Schema
{db_schema}
-----
Question: {question}
SQL:""".strip()

ANSWER_GEN_PROMPT = """
Answer the question based on the SQL execution results.
- The answer should be in JSON format of either a single value or a list of values.
- Output only the answer without additional explanation.
-----
### SQL
{sql}

### SQL Results
{sql_result}
-----
Question: {question}
Answer:""".strip()


def parse_sql(sql):
    sqls = re.findall(r"```sql\n(.*?)\n```", sql, re.DOTALL)
    return sqls[-1] if sqls else None


def process(query_row, llm_name):
    t0 = time.time()

    question = query_row["Query"]
    db_name = query_row["DB used"]
    db_path = f'../dev_folder/dev_databases/{db_name}/{db_name}.sqlite'
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    llm = ChatOpenAI(model=llm_name, temperature=0.0)
    write_query = create_sql_query_chain(llm, db) | parse_sql
    execute_query = QuerySQLDatabaseTool(db=db)
    chain = RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    query = None
    prediction = None
    error = None
    try:
        resp = chain.invoke({"question": question})
        query = resp["query"]
        prediction = resp["result"]
        print(type(prediction))
        prediction = json.loads(prediction)
        prediction = [row[0] for row in prediction]
        if not isinstance(query_row["Answer"], list) and len(prediction) == 1:
            prediction = prediction[0]
        print(prediction)
    except Exception as e:
        traceback.print_exc()
        error = str(e)

    return {
        "query_id": query_row["Query ID"],
        "prediction": prediction,
        "answer": query_row["Answer"],
        "sql_statement": query,
        "sql_results": None,
        "latency": time.time() - t0,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", default="../tag_queries.csv", type=str)
    parser.add_argument("--llm", default='gpt-4o', type=str)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--output_dir", default='out_text2sql_agent/', type=str)
    args = parser.parse_args()
    print(args)
    print()

    queries_df = pd.read_csv(args.df_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # for i, row in queries_df.head(1).iterrows():
    #     process(row, args.llm)

    all_outputs = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process, row, args.llm): row for _, row in queries_df.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            # print(result)
            all_outputs.append(result)
            if args.output_dir:
                with open(os.path.join(args.output_dir, f"query_{result['query_id']}.json"), "w") as f:
                    json.dump(result, f)

    all_outputs.sort(key=lambda x: int(x["query_id"]))
    with open(os.path.join(args.output_dir, f"all_outputs.json"), "w") as f:
        json.dump(all_outputs, f, indent=2)

    eval(queries_df, args.output_dir)


if __name__ == "__main__":
    main()
