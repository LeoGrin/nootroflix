import streamlit as st
import extra_streamlit_components as stx
import datetime
import time
from train_model import predict, evaluate
from utils import save_new_ratings, generate_user_id, load_collection
from streamlit.report_thread import get_report_ctx
from new_names import weird_nootropics, classic_nootropics, lifestyle_nootropics, all_nootropics
import streamlit.components.v1 as components


st.set_page_config(page_title="️Nootroflix", page_icon=":brain:", layout="centered", initial_sidebar_state="auto", menu_items=None)

deployed = True

if deployed:
    collection_ratings, collection_users = load_collection()

session_id = get_report_ctx().session_id
cookie_manager = stx.CookieManager()



if "counter" not in st.session_state:
    st.session_state.counter = 1 #weird hack to allow scrolling to the top on refresh

if "already_run" not in st.session_state.keys():
    st.session_state.already_run = True
    cookie_manager.get("userID")
else:
    user_id = cookie_manager.get("userID")
    print(user_id)
    if not user_id:
        #print("No username found, generating one...")
        user_id = generate_user_id("data/dataset_clean_right_names.csv", session_id)
        #print("UserID: {}".format(user_id))
        cookie_manager.set("userID", user_id, expires_at=datetime.datetime(year=2050, month=2, day=2))
        #print("cookie set")


st.title(':brain: Nootroflix')
original_title = '<p style="color:Pink; font-size: 20px;">Rate the nootropics you\'ve tried, and we\'ll tell you which one should work for you!</p>'
st.markdown(original_title, unsafe_allow_html=True)
#col1, col2 = st.columns([1, 1.25])
#with col1:
#    st.markdown("**Your results await at the bottom of the page **")
#with col2:
#    st.image("images/arrow.png", width=25)
#st.text("")

#if st.button("More infos"):
#    st.write("Our algorithm matches you to people with similar ratings, and tells you other nootropics they liked.")
#    st.write("The initial data comes from the 2016 SlateStarCodex Nootropics survey results.")
#    st.write("Some of the question are inspired by the 2016 and 2020 SlateStarCodex nootropics surveys.")

if "mode" not in st.session_state.keys():
    st.session_state["mode"] = "selection"


def go_to_mode_rating():
    for key in st.session_state.keys():
        if key.startswith("checkbox"):
            if st.session_state[key]:
                return go_to_mode("rating")()

    st.warning("Please rate at least one nootropic")

def go_to_mode(mode):
    def callback_mode():
        st.session_state["mode"] = mode
        for key in st.session_state.keys():
            if not key.startswith("permanent"):
                st.session_state["permanent_" + key] = st.session_state[key]
        st.session_state.counter += 1
    return callback_mode

def retrieve_widget_value(key):
    #allows to retrieve values chosen for going back
    #the reason is session states linked to widget disappear on reruns
    if "permanent_{}".format(key) in st.session_state.keys():
        st.session_state[key] = st.session_state["permanent_{}".format(key)]

def reset_selection():
    for key in st.session_state.keys():
        if key.startswith("checkbox") or key.startswith("permanent_checkbox"):
            st.session_state[key] = False


if st.session_state["mode"] == "selection":
    st.header("How do I use it?")
    st.markdown(""" **First tell us which nootropics you have tried, then rate your subjective experience on a scale of 0 to 10.**""")

    st.button("Reset 🗑", on_click=reset_selection)
    select_form = st.form("select-form")
    with select_form:
        n_cols = 2 #TODO align columns ?
        st.header("🧠 Classic nootropics")
        cols_classic = st.columns(n_cols)
        for i, nootropic in enumerate(classic_nootropics):
            with cols_classic[i % n_cols]:
                retrieve_widget_value("checkbox_{}".format(nootropic))
                st.checkbox("I've tried {}".format(nootropic), key="checkbox_{}".format(nootropic))
        st.write("")
        st.header("🧠 Other nootropics")
        cols_others = st.columns(n_cols)
        for i, nootropic in enumerate(weird_nootropics):
            with cols_others[i % n_cols]:
                retrieve_widget_value("checkbox_{}".format(nootropic))
                st.checkbox("I've tried {}".format(nootropic), key="checkbox_{}".format(nootropic))
        st.write("")
        st.header("🧠 Lifestyle")
        cols_lifestyle = st.columns(n_cols)
        for i, nootropic in enumerate(lifestyle_nootropics):
            with cols_lifestyle[i % n_cols]:
                retrieve_widget_value("checkbox_{}".format(nootropic))
                st.checkbox("I've tried {}".format(nootropic), key="checkbox_{}".format(nootropic))
        st.write("")
        st.form_submit_button("Next", on_click=go_to_mode_rating)

if st.session_state["mode"] == "rating":
    st.subheader("Please rate your experience with each nootropic.")
    st.markdown("""    \n 🧠 0 means a substance was totally useless, or had so many side effects you couldn't continue taking it.
    \n 🧠 1 - 4 means subtle effects, maybe placebo but still useful.
    \n 🧠 5 - 9 means strong effects, definitely not placebo.
    \n 🧠 10 means life - changing.""")
    st.write("")
    st.write("")

    #checkbox_dic = st.session_state["checkbox_dic"]
    # cols = st.columns(4)
    # i = 0
    # for nootropic in all_nootropics:
    #     if st.session_state["permanent_checkbox_{}".format(nootropic)]:
    #         with cols[i % 4]:
    #             st.button("{} 🗑".format(nootropic))
    #         i+=1
    possible_issues_list = ["None / Unsure",
                            "I developed tolerance",
                            "I developed addiction",
                            "I had to stop because of side effects",
                            "... and they persisted for some time after cessation",
                            "Other issues"]

    rating_form = st.form("rating-form")
    with rating_form:
        cols = st.columns(2)
        i = 0
        for nootropic in all_nootropics:
                if st.session_state["permanent_checkbox_{}".format(nootropic)]:
                    with cols[i%2]:
                        i += 1
                        retrieve_widget_value("slider_{}".format(nootropic))
                        st.slider("{} rating".format(nootropic), min_value=0, max_value=10, key="slider_{}".format(nootropic))
                        retrieve_widget_value("radio_{}".format(nootropic))
                        st.selectbox("Issues with {}".format(nootropic), possible_issues_list, key="radio_{}".format(nootropic))
                        st.write("")
                        st.write("")
        st.write("")
        col1, col2 = st.columns([1, 9])
        with col1:
            st.form_submit_button("Back", on_click=go_to_mode("selection")) #TODO save results when going back
        with col2:
            st.form_submit_button("Next", on_click=go_to_mode("questions"))

# for i, nootropic in enumerate(classic_nootropics):
#     checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
#     if checkbox_dic[nootropic]:
#         slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)
#         radio_dic[nootropic] = st.selectbox("Issues with {}".format(nootropic), possible_issues_list)
# st.write("")
# st.header("🧠 Other nootropics")
# for i, nootropic in enumerate(weird_nootropics):
#     checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
#     if checkbox_dic[nootropic]:
#         slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)
#         radio_dic[nootropic] = st.selectbox("Issues with {}".format(nootropic), possible_issues_list)
# st.header("🧠 Lifestyle")
# st.caption("Please rate cognitive improvement only")
# for i, nootropic in enumerate(lifestyle_nootropics):
#     checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
#     if checkbox_dic[nootropic]:
#         slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)
#         radio_dic[nootropic] = st.selectbox("Issues with {}".format(nootropic), possible_issues_list)
if st.session_state["mode"] == "questions":
    question_form = st.form("question-form")
    with question_form:
        #st.form_submit_button("Back", on_click=go_to_mode("ratings"))
        st.header("🧠 A few questions")
        st.selectbox("Gender", ["-", "Male", "Female", "Other"], key="question_gender")
        st.number_input("Age", min_value=0, max_value=100, value=0, key="question_age")
        options = ["Not at all a reason", "Yes, a minor reason", "Yes, a major reason"]
        st.radio("Do you take nootropics to help with anxiety?", options=options, key="question_anxiety")
        st.radio("Do you take nootropics to help with focus?", options=options, key="question_focus")
        st.radio("Do you take nootropics to help with mood?", options=options, key="question_mood")
        st.radio("Do you take nootropics to help with cognition / memory?", options=options, key="question_cognition")
        st.radio("Do you take nootropics to help with motivation?", options=options, key="question_motivation")
        st.write("")
        favorite_noot = st.text_input("What is your favorite nootropics not mentioned here?")
        st.text("")
        st.checkbox("Check this box if you're not entering your true ratings / infos (prevents training on your data)", key="not_true_ratings")
        st.text("")
        col1, col2 = st.columns([1, 9])
        with col1:
            st.form_submit_button("Back", on_click=go_to_mode("rating")) #TODO save results when going back
        with col2:
            st.form_submit_button("Get results!", on_click=go_to_mode("results"))


if st.session_state["mode"] == "results":
    slider_dic = {}
    radio_dic = {}
    question_dic = {}
    for key in st.session_state.keys():
        if key.startswith("permanent_slider"):
            slider_dic[key[len("permanent_slider_"):]] = st.session_state[key]
        elif key.startswith("permanent_radio_"):
            radio_dic[key[len("permanent_radio_"):]] = st.session_state[key]
        elif key.startswith("permanent_question_"):
            question_dic[key[len("permanent_question_"):]] = st.session_state[key]

    #pseudo = st.text_input("Pseudo")
    pseudo = "default"

    def left_align(s, props='text-align: left;'): #to left align values in dataframe
        return props

    st.button("Back", on_click=go_to_mode("questions"))
    new_result_df = predict(slider_dic)
    st.header("🧠 Your results")
    st.write("Our model predicted these ratings for you:")
    st.table(new_result_df.set_index("nootropic").style.format("{:.1f}").applymap(left_align))
    if not st.session_state["permanent_not_true_ratings"]:
        #print("saving...")
        if deployed:
            save_new_ratings(rating_dic=slider_dic,
                         issues_dic = radio_dic,
                         question_dic = question_dic,
                         is_true_ratings=not st.session_state["permanent_not_true_ratings"],
                         accuracy_check=False,
                         user_id=user_id,
                         pseudo = pseudo,
                         time = time.time(),
                         collection_ratings=collection_ratings,
                         collection_users=collection_users)
    st.header("🧠 How accurate is our model?")
    if len(slider_dic) < 2:
        st.warning("Please rate more nootropics")
    else:
        accuracy_df = evaluate(slider_dic)
        if not accuracy_df is None:
            st.write("For each nootropic, we hid your rating to our model, and had the model try to guess it.")
            st.caption("Some nootropics don't have enough data right now to be included.")
            st.table(accuracy_df.set_index("nootropic").style.format("{:.1f}").applymap(left_align))
            #print("saving...")
            if deployed:
                save_new_ratings(rating_dic=slider_dic,
                             issues_dic=radio_dic,
                             question_dic=question_dic,
                             is_true_ratings=not st.session_state["permanent_not_true_ratings"],
                             accuracy_check=True,
                             user_id=user_id,
                             pseudo = pseudo,
                             time=time.time(),
                             collection_ratings=collection_ratings,
                             collection_users=collection_users)

    st.button("Back", on_click=go_to_mode("questions"), key="results_2")
    st.button("Start again", on_click=go_to_mode("selection"))




components.html(
    f"""
        <p>{st.session_state.counter}</p>
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """,
    height=0
)