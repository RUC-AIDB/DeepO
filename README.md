<img src="./docs/logo_black.png" width="200">

## Source code for DeepO, A deep-learning-based query optimizer.

### Introduction

DeepO is a learning-based query optimizer that offers  high-quality and fine-grained optimization to user queries. We implement DeepO and incorporate it into [PostgreSQL](https://www.postgresql.org/ftp/source/v12.4/), and we also provide with a [web UI](https://github.com/RUC-AIDB/DeepO.Demo), where users can carry out the optimization operations interactively and evaluate the optimization performance.

---

### How to Setup
#### Requirements
- PostgreSQL installed with [pg_hint_plan](https://pghintplan.osdn.jp/pg_hint_plan.html) extension.
- [JOB](https://github.com/concretevitamin/join-order-benchmark) dataset downloaded and loaded into PostgreSQL.
- Create the virtual environment and activate it
  ```
  conda env create -f environment.yml
  conda activate deepo
  ```

#### System configuration
- Set the connection configuration of PostgreSQL in ```./src/get_plan```

#### Run DeepO

- Use following commands if you want to train your own Cost Learner.
    ```
    # Generate scan embedding intermediate result
    python scan_embedding.py
    # embedding plan into sequence
    python plan_to_seq.py
    # train the Cost Learner
    python cost_learning.py
    # evaluate the learning performance
    python cost_evaluation.py
    ```
- Use following commands to estimate cost and optimize new queries.
    ```
    python cost_estimation.py
    python hint_generation.py
    # The optimized queries will be saved in /data/SQL_with_hint
    ```
---
### Questions
This repository is still under development, contact Luming Sun (sunluming@ruc.edu.cn) if you have any questions.