import argparse
import json
import os
import re
import sqlite3
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from collections import defaultdict
import numpy as np
from litellm import batch_completion

from lotus.models import OpenAIModel

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", default="../tag_queries.csv", type=str)
    parser.add_argument("--llm", default='gpt-4o-mini', type=str)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--output_dir", default='out_text2sql_yanlin/', type=str)
    args = parser.parse_args()
    print(args)
    print()

    queries_df = pd.read_csv(args.df_path)
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    for i in range(0, len(queries_df), args.batch_size):
        j = min(i + args.batch_size, len(queries_df))
        text2sql_prompts = []
        for k in range(i, j):
            row = queries_df.iloc[k]
            question = row["Query"]
            db_schema = row["text2sql_prompt"]
            assert '-- Using valid SQLite' in db_schema, "Invalid schema"
            db_schema = db_schema.split('-- Using valid SQLite')[0].strip()
            text2sql_prompts.append(TEXT2SQL_PROMPT.format(db_schema=db_schema, question=question))

        responses = batch_completion(
            model=args.llm,
            messages=[
                [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ] for prompt in text2sql_prompts
            ]
        )
        responses = [r.choices[0].message.content for r in responses]
        sqls = []
        print()
        break
    exit(9)

    question = query_row["Query"]
    db_name = query_row["DB used"]
    messages = [[{"role": "user", "content": text2sql_prompt}]]
    raw_answer = lm(messages)[0]
    sql_statements = re.findall(r"```sql\n(.*?)\n```", raw_answer, re.DOTALL)
    if not sql_statements:
        sql_statements = re.findall(r"```\n(.*?)\n```", raw_answer, re.DOTALL)
    if not sql_statements:
        sql_statements = [raw_answer]

    last_sql_statement = sql_statements[-1]
    try:
        try:
            answer = eval(query_row["Answer"])
        except Exception:
            answer = query_row["Answer"]

        conn = sqlite3.connect(f"../dev_folder/dev_databases/{db_name}/{db_name}.sqlite")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        assert last_sql_statement.lower().startswith("select"), "Only SELECT queries are supported"
        cursor.execute(last_sql_statement)
        raw_results = cursor.fetchall()
        row_dicts = []
        for result in raw_results:
            row_dicts.append({col: result[col] for col in result.keys()})

        sql_result = ""
        for i, row in enumerate(row_dicts):
            sql_result += f"Row {i + 1}\n{row_to_str(row)}\n\n"
            if i >= 1000:
                break

        prompt = ANSWER_GEN_PROMPT.format(
            sql=last_sql_statement,
            sql_result=sql_result,
            question=question
        )
        # print(prompt)

        messages = [[{"role": "user", "content": prompt}]]
        prediction = lm(messages)[0]
        prediction = prediction.replace('```json', '').replace('```', '').strip()

        if isinstance(answer, list) and not isinstance(prediction, list):
            prediction = [prediction]
        elif not isinstance(answer, list) and isinstance(prediction, list):
            prediction = prediction[0]

        try:
            prediction = json.loads(prediction)
        except Exception:
            prediction = prediction

        return {
            "query_id": query_row["Query ID"],
            "prediction": prediction,
            "answer": answer,
            "sql_statement": last_sql_statement,
            "sql_results": json.dumps(row_dicts),
            "latency": time.time() - t0,
        }

    except Exception as e:
        return {
            "error": f"Error running SQL statement: {last_sql_statement}\n{e}",
            "query_id": query_row["Query ID"],
            "latency": time.time() - t0,
        }

    all_outputs = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_row, row): row for _, row in queries_df.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            print(result)
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
