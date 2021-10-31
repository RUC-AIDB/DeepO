import psycopg2

TIME_OUT = 60000

def get_query_plan(query,save_path):
    conn = psycopg2.connect(dbname="job", user="sunluming",host="127.0.0.1")

    cur = conn.cursor()
    cur.execute("SET statement_timeout = {};".format(TIME_OUT))
    cur.execute("LOAD 'pg_hint_plan';")
    try:
        cur.execute(query)
        result = cur.fetchall()
        plan = "\n".join(each[0] for each in result)

    except psycopg2.Error as e:
        plan = e.pgerror

    
    
    with open(save_path,"w") as f:
        f.writelines(plan)

    return plan