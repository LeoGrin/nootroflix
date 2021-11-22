import pandas as pd
from google.cloud import firestore
from google.cloud import firestore
from google.oauth2 import service_account
import os

#key_dict = json.loads(st.secrets["textkey"])
#for heroku
keys = ["project_id", "type", "private_key_id", "private_key",
        "client_email", "client_id", "auth_uri", "token_uri",
        "auth_provider_x509_cert_url", "client_x509_cert_url"]
cred_dic = {}
for key in keys:
    cred_dic[key] = os.environ.get(key).replace("\\n", "\n")

creds = service_account.Credentials.from_service_account_info(cred_dic)
db = firestore.Client(credentials=creds, project="nootropics-2a049")
users = list(db.collection(u'ratings').stream())

users_dict = list(map(lambda x: x.to_dict(), users))
df = pd.DataFrame(users_dict)

df.to_csv("test.csv")