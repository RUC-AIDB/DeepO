from get_plan import get_query_plan
from tqdm import tqdm

with open("../example/SQL_with_hint/0","r") as f:
    queries_with_hint = f.readlines()

for idx in tqdm(range(318,len(queries_with_hint))):
    plan = get_query_plan(queries_with_hint[idx],save_path="../example/optimized plans/{}".format(idx))
    # break