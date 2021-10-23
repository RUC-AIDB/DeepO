import psycopg2



conn = psycopg2.connect(dbname="job", user="sunluming",host="127.0.0.1")

cur = conn.cursor()

# query = "select * from aka_name"
query = "select * from aka_name, title where title.id = aka_name.id"
explain_query = "EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS) {};".format(query)
# explain_query = "EXPLAIN {};".format(query)

cur.execute(explain_query)

result = cur.fetchall()

# print(result)

plan = "\n".join(each[0] for each in result)
# print(plan)

with open("result.txt","w") as f:
    f.writelines(plan)
# print(records)