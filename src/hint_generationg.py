# 3 kinds of hints
# Hints for scan methods
# Hints for join methods
# Hint for joining order
# %%
import csv
import os
import numpy as np
from itertools import permutations
from get_plan import get_query_plan
# %%
def load_data(file_name):
    joins = []
    predicates = []
    tables = []
    label = []

    # Load queries
    with open(file_name, 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            # label.append(row[3])
    print("Loaded queries")

    return tables, joins, predicates

# %%
SQL_PATH = "/home/sunluming/download/learnedcardinalities/data/train.csv"
tables, joins, predicates = load_data(SQL_PATH)

# %%
def cartesian(arrays, dtype=None, out=None):
    arrays = [np.asarray(x) for x in arrays]
    if dtype is None:
        dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out
# %%
def generate_scan_hints(tables):
    scan_methods = ["SeqScan({})", "IndexScan({})"]
    hint_candidate = []
    for table in tables:
        table_candidate =[]
        t = table.split(" ")[1]
        for method in scan_methods:
            table_candidate.append(method.format(t))
        hint_candidate.append(table_candidate)
    candidates = [" ".join(x) for x in cartesian(hint_candidate, 'object')]
    return candidates
# %%
def generate_join_method_hints(tables):
    join_methods = ["NestLoop({})","MergeJoin({})","HashJoin({})"]
    if(tables==[""]):
        return [""]
    hint_candidate = []
    for join in tables:
        join_table = [x.split(".")[0] for x in join.split("=")]
        join_candidate = [each.format(" ".join(join_table)) for each in join_methods]
        hint_candidate.append(join_candidate)
    candidates = [" ".join(x) for x in cartesian(hint_candidate, 'object')]
    return candidates
# %%
def generate_join_order_hins(tables):
    if(len(tables)==1):
        return [""]
    join_tables = [x.split(" ")[1] for x in tables]
    orders = list(permutations(join_tables, len(join_tables)))
    join_orders = ["Leading ({})".format(" ".join(each)) for each in orders]
    return join_orders
# %%
def construct_sql(table, join, predicates):
    sql = "explain select count(*) from {} where {} and {}"
    tables = ", ".join(table)
    joins = " and ".join(join)
    l = []
    for n in range(len(predicates)//3):
        l.append(' '.join(predicates[n*3:n*3+3]))
    predicates = " and ".join(l)
    return sql.format(tables,joins,predicates)
# %%
def generate_hint_queries(query_idx):
    global tables
    global joins
    global predicates
    scan_hints = generate_scan_hints(tables[query_idx])
    join_method_hints = generate_join_method_hints(joins[query_idx])
    join_order_hints = generate_join_order_hins(tables[query_idx])
    candidates = [scan_hints, join_method_hints, join_order_hints]
    hints_set = [" ".join(x) for x in cartesian(candidates, 'object')]
    sql = construct_sql(tables[query_idx],joins[query_idx],predicates[query_idx])
    queries = []
    for each in hints_set:
        query = "LOAD 'pg_hint_plan';\
            /*+ {} */ ".format(each) + sql + ";"
        queries.append(query)
    return queries
# %%
# for i in range(len(tables)):
#     i = 9
#     scan_hints = generate_scan_hints(tables[i])
#     # print(scan_hints)
#     join_method_hints = generate_join_method_hints(joins[i])
#     # print(join_method_hints)
#     join_order_hints = generate_join_order_hins(tables[i])
#     # print(join_order_hints)
#     candidates = [scan_hints, join_method_hints, join_order_hints]
#     hints_set = [" ".join(x) for x in cartesian(candidates, 'object')]
#     # print(hints_set)
#     sql = construct_sql(tables[i],joins[i],predicates[i])
#     # print(sql)
#     for each in hints_set:
#         query = "/*+ {} */ ".format(each) + sql + ";"
#         print(query)
#     break
# %%
query_idx = 9
queries_with_hint = generate_hint_queries(query_idx)
os.makedirs("../data/plan/{}".format(query_idx), mode=0o777, exist_ok=True)

for idx,query in enumerate(queries_with_hint):
    plan = get_query_plan(query,save_path="../data/plan/{}/{}".format(query_idx,idx))
    print(plan)
    # break
# %%