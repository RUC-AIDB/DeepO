# %%
import numpy as np
import pandas as pd
import os

# %%
optimized_plan_files = os.listdir("../example/optimized plans")

raw_plan = "../example/raw.plan"
raw_cost = 9159.697



# %%
optimized_costs = {}
optimizaed_plans = {}
for file in optimized_plan_files:
    with open(os.path.join("../example/optimized plans",file),"r") as f:
        operators = []
        optimized_plan = f.readlines()
        for line in optimized_plan:
            if("->" in line):
                operators.append(line.split("->")[1].split("  ")[0])
        optimizaed_plans[file] = operators
        # print(optimized_plan)
        optimized_cost = float(optimized_plan[0].split("  ")[1].split("..")[-1].split(" ")[0])
        optimized_costs[file] = optimized_cost
        # break
# %%
optimized_costs
# %%
pd.DataFrame((list(optimized_costs.values()))).hist()
# %%
test_write