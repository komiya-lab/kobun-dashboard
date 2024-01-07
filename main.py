import streamlit as st
import os

st.write("# Kobun Dashboard")

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PJDIR = os.path.join(BASEDIR, "projects")

with st.form("create", True):
    name = st.text_input("Project Name")
    sbm = st.form_submit_button()

pj_dir = os.path.join(PJDIR, name)

if sbm:
    if os.path.exists(pj_dir):
        st.toast(f"Error: The project named {name} exists.")

    else:
        os.makedirs(pj_dir)
        st.toast(f"The project named {name} is created.")
