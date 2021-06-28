#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader

from train_model import predict, evaluate

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

nootropics_list = rating_example.keys()

st.title('Nootropics recommandation system (MVP)')
st.warning(
    "Tell us which nootropics you have tried. For each substance, please rate your subjective experience on a scale of 0 to 10. 0 means a substance was totally useless, or had so many side effects you couldn't continue taking it. 1 - 4 means for subtle effects, maybe placebo but still useful. 5 - 9 means strong effects, definitely not placebo. 10 means life-changing.")

slider_dic = {}
checkbox_dic = {}
for nootropic in nootropics_list:
    checkbox_dic[nootropic] = st.checkbox("I've tried {}".format(nootropic))
    if checkbox_dic[nootropic]:
        slider_dic[nootropic] = st.slider("{} rating".format(nootropic), min_value=0, max_value=10)
    # form = st.form(key=nootropic)
    # form.text_input(label="{} rating".format(nootropic))
    # form_dic[nootropic] = form
    # submit_button = st.form_submit_button(label='Submit')
print(checkbox_dic)
print(slider_dic)

if st.button("I'm done rating and would like to see predictions"):
    new_result_df = predict(slider_dic)
    st.write("Our model predicted this ratings for you:")
    st.write(new_result_df)

if st.button("How accurate is your model ?"):
    if len(slider_dic) < 2:
        st.error("Please rate at least two nootropics")
    else:
        accuracy_df = evaluate(slider_dic)
        st.write("For each nootropic, we hid your rating to our model, and had the model try to guess it.")
        st.write(accuracy_df)