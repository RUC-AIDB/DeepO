# 3 kinds of hints
# Hints for scan methods
# Hints for join methods
# Hint for joining order
# %%
import csv
import os
import numpy as np
import copy
from itertools import permutations
from tqdm import tqdm
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
# SQL_PATH = "/home/sunluming/demo/example/example.sql"
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
def add_one_rel(cur, join_tables):
    extended_order = []
    for table in join_tables:
        if(table not in cur):
            tmp = ["("]
            tmp.extend(cur)
            tmp.append(table)
            tmp.append(")")
            extended_order.append(tmp)

            tmp = ["("]
            tmp.append(table)            
            tmp.extend(cur)
            tmp.append(")")

            extended_order.append(tmp)
        else:
            continue
    return extended_order
   
def generate_join_order_hins(tables):
    if(len(tables)==1):
        return [""],[]
    join_tables = [x.split(" ")[1] for x in tables]
    num_tables = len(tables)
    str_order_length = 3*num_tables-2
    join_orders = []
    starter = copy.deepcopy(join_tables)
    stack = [[each] for each in starter]
    while(len(stack)!=0):
        cur = stack.pop(0)
        if(len(cur)<str_order_length):
            extended_orders = add_one_rel(cur, join_tables)
            stack.extend(extended_orders)
        else:
            join_orders.append(cur)
    str_join_orders = [" ".join(each) for each in join_orders]
    # print(str_join_orders)
    str_join_orders = set(str_join_orders)
    join_orders_string = ["Leading ({})".format(each) for each in str_join_orders]
    # print(join_orders)
    return join_orders_string, join_orders
# %%
def construct_sql(table, join, predicates,method="explain"):
    tables = ", ".join(table)
    print(join)
    print(predicates)
    if(join!=[""] and predicates!=[""]):
        joins = " and ".join(join)
        sql = method + " select count(*) from {} where {} and {}"
    elif(join!=[""] and predicates==[""]):
        joins = " and ".join(join)
        sql = method + " select count(*) from {} where {} {}"
    elif(join==[""] and predicates!=[""]):
        joins = ""
        sql = method + " select count(*) from {} where {} {}"
    else:
        joins = ""
        sql = method + " select count(*) from {} {} {}"
    l = []
    for n in range(len(predicates)//3):
        l.append(' '.join(predicates[n*3:n*3+3]))
    predicates = " and ".join(l)
    return sql.format(tables,joins,predicates)

# %%
def parse_order(order):
    left = 0
    right = len(order) - 1
    parsed_order = []
    while(left<right):
        if(order[left]=="(" and order[right]==")"):
            left += 1
            right -= 1
        elif(order[left]=="("):
            parsed_order.insert(0,order[right])
            right -= 1
        elif(order[right]==")"):
            parsed_order.insert(0,order[left])
            left += 1
        else:
            parsed_order.insert(0,order[right])
            parsed_order.insert(0,order[left])
            left += 1
            right -= 1
    return parsed_order


def generate_join_method_hints_from_orders(join_order_hints, join_orders_list):
    join_methods = ["NestLoop({})","MergeJoin({})","HashJoin({})"]
    join_hints = []
    for order_hint, order in zip(join_order_hints,join_orders_list):
        parsed_order = parse_order(order)
        join_order = []
        for idx in range(2,len(parsed_order)+1):
            join_order.append(" ".join(parsed_order[0:idx]))
        join_candidate = []
        for idx,level in enumerate(join_order):
            join_candidate.append([each.format(level) for each in join_methods])
        candidates = [" ".join(x) for x in cartesian(join_candidate, 'object')]
        join_hints.extend([each + " " + order_hint for each in candidates])
    if(join_hints==[]):
        join_hints = [""]
    return join_hints
# %%
def generate_hint_queries(query_idx,method):
    global tables
    global joins
    global predicates
    scan_hints = generate_scan_hints(tables[query_idx])
    # join_method_hints = generate_join_method_hints(joins[query_idx])
    join_order_hints, join_orders = generate_join_order_hins(tables[query_idx])
    join_hints = generate_join_method_hints_from_orders(join_order_hints,join_orders)
    candidates = [scan_hints, join_hints]
    hints_set = [" ".join(x) for x in cartesian(candidates, 'object')]
    sql = construct_sql(tables[query_idx],joins[query_idx],predicates[query_idx],method)
    queries = []
    for each in hints_set:
        query = "/*+ {} */ ".format(each) + sql + ";"
        queries.append(query)
    return queries, sql+";"
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
for query_idx in tqdm(range(0,20)):
    # query_idx = 9
    queries_with_hint, sql = generate_hint_queries(query_idx, method="explain analyse")
    os.makedirs("../data/plan/{}".format(query_idx), mode=0o777, exist_ok=True)
    os.makedirs("../data/SQL/".format(query_idx), mode=0o777, exist_ok=True)
    os.makedirs("../data/SQL_with_hint/".format(query_idx), mode=0o777, exist_ok=True)

    with open("../data/SQL/{}".format(query_idx),"w") as f:
        f.writelines(sql)

    with open("../data/SQL_with_hint/{}".format(query_idx),"w") as f:
        f.writelines("\n".join(queries_with_hint))


    for idx,query in enumerate(tqdm(queries_with_hint)):
        plan = get_query_plan(query,save_path="../data/plan/{}/{}".format(query_idx,idx))
        
    # print(plan)
    # break
# %%
# for example
# for query_idx in tqdm(range(0,1)):
#     # query_idx = 9
#     queries_with_hint, sql = generate_hint_queries(query_idx, method="explain analyse")
#     os.makedirs("../example/SQL/".format(query_idx), mode=0o777, exist_ok=True)
#     os.makedirs("../example/SQL_with_hint/".format(query_idx), mode=0o777, exist_ok=True)

#     with open("../example/SQL/{}".format(query_idx),"w") as f:
#         f.writelines(sql)

#     with open("../example/SQL_with_hint/{}".format(query_idx),"w") as f:
#         f.writelines("\n".join(queries_with_hint))


#     for idx,query in enumerate(tqdm(queries_with_hint)):
#         plan = get_query_plan(query,save_path="../example/optimized plans/{}".format(idx))
        
# %%
