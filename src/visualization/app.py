# src/visualization/app.py
"""
Streamlit dashboard for FaceStats v3.5
"""
import streamlit as st


def main():
    st.title("FaceStats v3.5 Dashboard")
    st.write("Upload an image → get full demographic + embedding + attractiveness stats.")

    # TODO: plug in UserUploadScorer


if __name__ == "__main__":
    main()
