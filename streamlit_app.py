import streamlit as st
import extra_streamlit_components as stx
import datetime
import time
from train_model import predict, evaluate
from utils import save_new_ratings, generate_user_id, load_collection
from streamlit.report_thread import get_report_ctx
from new_names import weird_nootropics, classic_nootropics, lifestyle_nootropics



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

st.set_page_config(page_title="️Nootroflix", page_icon=":brain:", layout="centered", initial_sidebar_state="auto", menu_items=None)

collection_ratings, collection_users = load_collection()

nootropics_list = list(rating_example.keys())

nootropics_list.sort()


session_id = get_report_ctx().session_id
cookie_manager = stx.CookieManager()

if "already_run" not in st.session_state.keys():
   st.session_state.already_run = True
   cookie_manager.get("userID")
else:
   user_id = cookie_manager.get("userID")
   print(user_id)
   if not user_id:
       print("No username found, generating one...")
       user_id = generate_user_id("data/dataset_clean.csv", session_id)
       print("UserID: {}".format(user_id))
       cookie_manager.set("userID", user_id, expires_at=datetime.datetime(year=2050, month=2, day=2))
       print("cookie set")


st.title('Nootroflix')
#st.info("")
original_title = '<p style="color:Pink; font-size: 20px;">Rate the nootropics you\'ve tried, and we\'ll tell you which one should work for you!</p>'
st.markdown(original_title, unsafe_allow_html=True)
#st.error("****")
st.header("🧠 How do I use it?")
st.markdown(""" **Tell us which nootropics you have tried, and rate your subjective experience on a scale of 0 to 10.**
\n 🧠 0 means a substance was totally useless, or had so many side effects you couldn't continue taking it.
\n 🧠 1 - 4 means subtle effects, maybe placebo but still useful.
\n 🧠 5 - 9 means strong effects, definitely not placebo.
\n 🧠 10 means life-changing.""")

# st.markdown(""" **Tell us which nootropics you have tried, and rate your subjective experience on a scale of 0 to 10.**
# \n 🧠 0 - 4 means the nootropic effect was negative.
# \n 🧠 5 - 7 means slight positive reaction, maybe placebo but still useful
# \n 🧠 8 - 9 means strong effects, definitely not placebo.
# \n 🧠 10 means life-changing.""")
col1, col2 = st.columns([1, 1.25])
with col1:
    st.markdown("**Your results await at the bottom of the page **")
with col2:
    st.image("arrow.png", width=25)
st.text("")
#col1, col2 = st.columns(2)
#col_list = [col1, col2]
slider_dic = {}
radio_dic = {}
checkbox_dic = {}
st.header("🧠 Classic nootropics")
possible_issues_list = ["None / Unsure",
                        "I developed tolerance",
                        "I developed addiction",
                        "I had to stop taking it because of side effects",
                        "I had to stop taking it because of side effects, and the side effects persisted for some time after cessation",
                        "Other issues"]
for i, nootropic in enumerate(classic_nootropics):
     #with col_list[i >= len(nootropics_list) / 2]:
    checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
    if checkbox_dic[nootropic]:
        slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)
        radio_dic[nootropic] = st.selectbox("Issues with {}".format(nootropic), possible_issues_list)
st.write("")
st.header("🧠 Other nootropics")
for i, nootropic in enumerate(weird_nootropics):
     #with col_list[i >= len(nootropics_list) / 2]:
    checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
    if checkbox_dic[nootropic]:
        slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)
        radio_dic[nootropic] = st.selectbox("Issues with {}".format(nootropic), possible_issues_list)
st.header("🧠 Lifestyle")
st.caption("Please rate cognitive improvement only")
for i, nootropic in enumerate(lifestyle_nootropics):
     #with col_list[i >= len(nootropics_list) / 2]:
    checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
    if checkbox_dic[nootropic]:
        slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)
        radio_dic[nootropic] = st.selectbox("Issues with {}".format(nootropic), possible_issues_list)
st.header("🧠 A few questions")
question_dic = {}
question_dic["gender"] = st.selectbox("Gender", ["-", "Male", "Female", "Other"])
question_dic["age"] = st.number_input("Age", min_value=0, max_value=100, value=0)
question_dic["for_anxiety"] = st.radio("Do you take nootropics to help with anxiety?", options=["Not at all a reason", "Yes, a minor reason", "Yes, a major reason"])
question_dic["for_focus"] = st.radio("Do you take nootropics to help with focus?", options=["Not at all a reason", "Yes, a minor reason", "Yes, a major reason"])
question_dic["for_mood"] = st.radio("Do you take nootropics to help with mood?", options=["Not at all a reason", "Yes, a minor reason", "Yes, a major reason"])
question_dic["for_cognition"] = st.radio("Do you take nootropics to help with cognition / memory?", options=["Not at all a reason", "Yes, a minor reason", "Yes, a major reason"])
question_dic["for_motivation"] = st.radio("Do you take nootropics to help with motivation?", options=["Not at all a reason", "Yes, a minor reason", "Yes, a major reason"])


favorite_noot = st.text_input("What is your favorite nootropics not mentioned here?")
if favorite_noot:
    slider_dic[favorite_noot] = st.slider("{} rating".format(favorite_noot), min_value=0, max_value=10)
st.text("")
st.text("")
st.header("🧠 Your results")
#pseudo = st.text_input("Pseudo")
pseudo = "default"
not_true_ratings = st.checkbox("Check this box if you're not entering your true ratings (prevents training on your data)")
if st.button("I'm done rating and would like to see predictions"):
    new_result_df = predict(slider_dic)
    st.write("Our model predicted these ratings for you:")
    st.write(new_result_df.set_index("nootropic").style.format("{:.2}"))
    #st.balloons()
    if not not_true_ratings:
        print("saving...")
        save_new_ratings(rating_dic=slider_dic,
                         issues_dic = radio_dic,
                         question_dic = question_dic,
                         is_true_ratings=not not_true_ratings,
                         accuracy_check=False,
                         user_id=user_id,
                         pseudo = pseudo,
                         time = time.time(),
                         collection_ratings=collection_ratings,
                         collection_users=collection_users)

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
                         issues_dic=radio_dic,
                         question_dic=question_dic,
                         is_true_ratings=not not_true_ratings,
                         accuracy_check=True,
                         user_id=user_id,
                         pseudo = pseudo,
                         time=time.time(),
                         collection_ratings=collection_ratings,
                         collection_users=collection_users)

if st.button("About"):
    st.write("Our algorithm matches you to people with similar ratings, and tells you other nootropics they liked.")
    st.write("The data comes from the 2016 SlateStarCodex Nootropics survey, [this Github](https://github.com/darktka/darktka.github.io), and data I got from redditor u/stupidlittlekids.")
    st.write("Some of the question are taken from 2016 and 2020 SlateStarCodex nootropics surveys.")
