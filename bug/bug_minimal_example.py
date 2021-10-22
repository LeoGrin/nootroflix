import streamlit as st
import extra_streamlit_components as stx
import time

cookie_manager = stx.CookieManager()
time.sleep(3)
print(1)
print(cookie_manager.get_all())

user_id = cookie_manager.get("userID")

if not user_id:
    print("here")
    user_id = 1000
    cookie_manager.set("userID", user_id)

print(2)
print(cookie_manager.get_all())