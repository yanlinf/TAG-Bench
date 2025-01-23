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

from lotus.models import OpenAIModel

from tag.utils import row_to_str, eval

ANSWER_GEN_PROMPT = """
Answer the question based on the SQL execution results.
- The answer should be in JSON format of either a single value or a list of values.
- Output only the answer without additional explanation.
---
SQL: {sql}

SQL Results: {sql_result}
---
Question: {question}
Answer:""".strip()


def run_row(query_row):
    t0 = time.time()
    text2sql_prompt = query_row["text2sql_prompt"]
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", default="../tag_queries.csv", type=str)
    parser.add_argument("--llm", default='gpt-4o-mini', type=str)
    parser.add_argument("--output_dir", default='output_text2sql_lm/', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    queries_df = pd.read_csv(args.df_path)
    lm = OpenAIModel(
        model=args.llm,
        api_base="https://api.openai.com/v1/",
        provider="openai",
        max_tokens=8192,
        temperature=0.0,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

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
