import argparse
import json
import os
import re
import sqlite3
import time
import traceback
from litellm import completion
import pandas as pd
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tag.utils import row_to_str, eval

AGENT_PROMPT = """Your task is to answer a question using a SQLite database. You are allowed to perform the following actions:
1. [SQL] Execute a SQL query to retrieve data from the database. The SQL should start with `SELECT`.

2. [ANSWER] Answer the question if there are enough evidence to do so.

Based on the past actions, determine the correct action to take at the current step.
The output should be in the format `[ACTION] <SQL or Answer>`. Do not include any additional explanation.
Example: `[SQL] SELECT * FROM table_name WHERE column_name = value`
-----
### SQLite DB Schema
{db_schema}

### Past actions
{past_actions}

### Question: {question}

### Action to take:
"""


class Text2SQLAgent:
    def __init__(self, sqlite_db_path, llm_name, max_steps: int = 10):
        self.db_path = sqlite_db_path
        self.max_steps = max_steps
        self.llm_name = llm_name
        self.conn = sqlite3.connect(self.db_path)
        self.db_schema = self.get_sqlite_schema()

    def get_sqlite_schema(self) -> str:
        """Extracts and linearizes the schema of an SQLite database."""
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL;")
        schema_statements = [row[0] for row in cursor.fetchall()]

        return "\n\n".join(schema_statements)

    def query(self, question: str):
        trajectory = []
        answer = None
        while True:
            if len(trajectory) >= self.max_steps:
                break

            past_actions = "\n".join(trajectory)
            prompt = AGENT_PROMPT.format(db_schema=self.db_schema, past_actions=past_actions, question=question)
            response = completion(
                model=self.llm_name,
                temperature=0.0,
                messages=[{"role": "system", "content": prompt}]
            ).choices[0].message.content

            action, content = response.split(" ", 1)
            action = action[1:-1]

            if action == "SQL":
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(content)
                    results = cursor.fetchall()
                    trajectory.append(json.dumps({'sql': content, 'results': results}))
                except Exception as e:
                    trajectory.append(json.dumps({'sql': content, 'error': str(e)}))

            elif action == "ANSWER":
                answer = content

        return answer, trajectory

    def close(self):
        self.conn.close()


def process(query_row, llm_name):
    t0 = time.time()

    question = query_row["Query"]
    db_name = query_row["DB used"]
    db_path = f'../dev_folder/dev_databases/{db_name}/{db_name}.sqlite'

    agent = Text2SQLAgent(db_path, llm_name)
    prediction, trajectory = agent.query(question)
    agent.close()

    return {
        "query_id": query_row["Query ID"],
        "question": question,
        "prediction": prediction,
        "answer": query_row["Answer"],
        "trajectory": trajectory,
        "latency": time.time() - t0
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", default="../tag_queries.csv", type=str)
    parser.add_argument("--llm", default='gpt-4o', type=str)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", default='out_text2sql_agent/', type=str)
    args = parser.parse_args()
    print(args)
    print()

    queries_df = pd.read_csv(args.df_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # for i, row in queries_df.head(1).iterrows():
    #     process(row, args.llm)

    if args.debug:
        result = process(queries_df.iloc[0], args.llm)
        print(result)
        return

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
