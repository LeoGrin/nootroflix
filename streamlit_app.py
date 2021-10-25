import streamlit as st
import extra_streamlit_components as stx
import datetime
import time
from train_model import predict, evaluate
from utils import save_new_ratings, generate_user_id, load_collection


rating_example = {'Modafinil': 6,
'Caffeine': 6,
'Coluracetam': None,
'Phenylpiracetam': None,
'Theanine': 7,
'Noopept': None,
'Oxiracetam': None,
'Aniracetam': None,
'Rhodiola': None,
'Creatine': 4,
'Piracetam': None,
'Ashwagandha': None,
'Bacopa': None,
'Choline': None,
'DMAE': None,
'Fasoracetam': None,
'SemaxandNASemaxetc': None,
'SelankandNASelanketc': None,
'Inositol': None,
'Seligiline': None,
'AlphaBrainproprietaryblend': None,
'Cerebrolysin': None,
'Melatonin': 8,
'Uridine': None,
'Tianeptine': None,
'MethyleneBlue': None,
'Unifiram': None,
'PRL853': None,
'Emoxypine': None,
'Picamilon': None,
'Dihexa': None,
'Epicorasimmunebooster': None,
'LSD': 7,
'Adderall': 8,
"Phenibut": 6,
"Nicotine": 7}


collection = load_collection()

nootropics_list = rating_example.keys()



cookie_manager = stx.CookieManager()

if "already_run" not in st.session_state.keys():
   st.session_state.already_run = True
   cookie_manager.get("userID")
else:
   user_id = cookie_manager.get("userID")
   print(user_id)
   if not user_id:
       print("No username found, generating one...")
       user_id = generate_user_id("data/dataset_clean.csv")
       print("UserID: {}".format(user_id))
       cookie_manager.set("userID", user_id, expires_at=datetime.datetime(year=2050, month=2, day=2))
       print("cookie set")


st.title('Nootropics recommandation system')
#st.info("")
st.markdown(""" **Tell us which nootropics you have tried, and rate your subjective experience on a scale of 0 to 10.**
- 0 means a substance was totally useless, or had so many side effects you couldn't continue taking it.
- 1 - 4 means for subtle effects, maybe placebo but still useful.
- 5 - 9 means strong effects, definitely not placebo.
- 10 means life-changing.""")
st.text("")
col1, col2 = st.columns(2)
col_list = [col1, col2]
slider_dic = {}
checkbox_dic = {}
for i, nootropic in enumerate(nootropics_list):
    with col_list[i%2]:
        checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
        if checkbox_dic[nootropic]:
            slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)

    # form = st.form(key=nootropic)
    # form.text_input(label="{} rating".format(nootropic))
    # form_dic[nootropic] = form
    # submit_button = st.form_submit_button(label='Submit')
st.text("")
st.text("")
pseudo = st.text_input("Pseudo")
st.text("")
not_true_ratings = st.checkbox("Check this box if you're not entering your true ratings (prevents our model from training on your data)")
if st.button("I'm done rating and would like to see predictions"):
    new_result_df = predict(slider_dic)
    st.write("Our model predicted these ratings for you:")
    st.write(new_result_df)
    #st.balloons()
    if not not_true_ratings:
        print("saving...")
        save_new_ratings(rating_dic=slider_dic,
                         is_true_ratings=not not_true_ratings,
                         accuracy_check=False,
                         user_id=user_id,
                         pseudo = pseudo,
                         time = time.time(),
                         collection=collection)

if st.button("How accurate is our model ?"):
    if len(slider_dic) < 2:
        st.error("Please rate at least two nootropics")
    else:
        accuracy_df = evaluate(slider_dic)
        st.write("For each nootropic, we hid your rating to our model, and had the model try to guess it.")
        st.write(accuracy_df)
        #st.balloons()
        print("saving...")
        save_new_ratings(rating_dic=slider_dic,
                         is_true_ratings=not not_true_ratings,
                         accuracy_check=True,
                         user_id=user_id,
                         pseudo = pseudo,
                         time=time.time(),
                         collection=collection)

