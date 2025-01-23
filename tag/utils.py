import heapq
import pandas as pd
import numpy as np
import json
from collections import defaultdict


def is_equal(pred, ans):
    if isinstance(ans, list):
        return isinstance(pred, list) and sorted(pred) == sorted(ans)
    return str(pred) == str(ans)


def eval(df, output_dir):
    grouped = df.groupby("Query type")
    latencies = defaultdict(list)
    corrects = defaultdict(list)
    for group_name, group_df in grouped:
        for _, row in group_df.iterrows():
            qid = row["Query ID"]

            with open(f"{output_dir}/query_{qid}.json") as f:
                data = json.load(f)
                latencies[group_name].append(data["latency"])
                if data.get("error", None):
                    corrects[group_name].append(False)
                else:
                    # corrects[group_name].append(data["prediction"] == data["answer"])
                    corrects[group_name].append(is_equal(data["prediction"], data["answer"]))

    kr_grouped = df.groupby("Knowledge/Reasoning Type")
    for group_name, group_df in kr_grouped:
        for _, row in group_df.iterrows():
            qid = row["Query ID"]
            if row["Query type"] != "Aggregation":
                with open(f"{output_dir}/query_{qid}.json") as f:
                    data = json.load(f)
                    latencies[group_name].append(data["latency"])
                    if data.get("error", None):
                        corrects[group_name].append(False)
                    else:
                        # corrects[group_name].append(data["prediction"] == data["answer"])
                        corrects[group_name].append(is_equal(data["prediction"], data["answer"]))

    for k, v in latencies.items():
        print(f"Printing stats for {k}")
        print(f"Mean latency: {np.mean(v):.2f}")
        print(f"Avg. correct: {np.mean(corrects[k]):.2f}")

    group_corrects = []
    for k in 'Overall	Match	Comparison	Ranking	Aggregation	Knowledge	Reasoning'.split('\t'):
        print(k)
        if k == 'Overall':
            correct = np.mean([v for k, vs in corrects.items() for v in vs if k != 'Aggregation'])
        else:
            correct = np.mean(corrects[k])
        group_corrects.append(correct)
    print('\t'.join([f'{k:.2f}' for k in group_corrects]))


class IndexMerger:
    class HeapElement:
        def __init__(self, distance, row):
            self.distance = distance
            self.row = row
            self.reranker_score = None

        def __lt__(self, other):
            return self.distance < other.distance

    def __init__(self, db_table_pairs, model):
        self.model = model
        self.db_table_pairs = db_table_pairs
        self.all_dfs = []
        for db, table in db_table_pairs:
            self.all_dfs.append(pd.read_csv(f"../pandas_dfs/{db}/{table}.csv"))

    def __call__(self, query, k):
        heap = []
        for i, (db, table) in enumerate(self.db_table_pairs):
            self.model.load_index(f"../indexes/{db}/{table}")
            df = self.all_dfs[i]
            distances, idxs = self.model(query, k)
            idxs = idxs[0]
            distances = distances[0]

            for idx, distance in zip(idxs, distances):
                if len(heap) < k:
                    heapq.heappush(heap, self.HeapElement(distance, df.iloc[idx]))
                else:
                    heapq.heappushpop(heap, self.HeapElement(distance, df.iloc[idx]))

        return heap


# Serialization format from STaRK - https://arxiv.org/pdf/2404.13207
def row_to_str(row):
    return "\n".join([f"- {col}: {val}" for col, val in row.items()])
