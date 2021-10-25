import pandas as pd
from google.cloud import firestore
import json
from google.cloud import firestore
from google.oauth2 import service_account
import streamlit as st

key_dict = json.loads(st.secrets["textkey"])
creds = service_account.Credentials.from_service_account_info(key_dict)
db = firestore.Client(credentials=creds, project="nootropics-2a049")
users = list(db.collection(u'ratings').stream())

users_dict = list(map(lambda x: x.to_dict(), users))
df = pd.DataFrame(users_dict)

df.to_csv("test.csv")