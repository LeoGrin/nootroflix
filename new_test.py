from os import environ
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('mysql+pymysql://root:22272016577255@127.0.0.1:3306/pets', echo=True)


print(pd.read_sql("test", engine))

#jobs_df = pd.read_csv('data/new_database.csv')

#table_name = 'test'

#jobs_df.to_sql(
#    table_name,
#    engine,
#    if_exists='replace',
#    index=False,
#    chunksize=500
#)