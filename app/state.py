from __future__ import annotations
import streamlit as st
import pandas as pd

# Session keys and defaults
_KEYS = {
    "results": None,
    "df": pd.DataFrame(),
    "error_text": "",
    "assignment_preview": "",
}

def ensure_session_state():
    for k, v in _KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Setters / Getters
def set_results_df(results, df: pd.DataFrame):
    st.session_state.results = results
    st.session_state.df = df

def get_results():
    return st.session_state.get("results")

def get_df() -> pd.DataFrame:
    return st.session_state.get("df", pd.DataFrame())

def set_error(text: str):
    st.session_state.error_text = text

def get_error() -> str:
    return st.session_state.get("error_text", "")

def set_assignment_preview(text: str):
    st.session_state.assignment_preview = text

def get_assignment_preview() -> str:
    return st.session_state.get("assignment_preview", "")
