import streamlit as st
import extra_streamlit_components as stx

cookie_manager = stx.CookieManager()

if "already_run" not in st.session_state.keys():
   st.session_state.already_run = True
   cookie_manager.get("userID")
else:
   print(cookie_manager.get("userID"))

