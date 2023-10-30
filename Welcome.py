import webbrowser

import streamlit as st
from dstack.api import Client, ClientError

st.title("Welcome to LLM Weaver! 👋")
st.caption(
    "This app helps fine-tune and deploy LLMs via user interface and access them for inference."
)

if "welcome" not in st.session_state:
    st.session_state.client = Client.from_config()
    try:
        with st.spinner("Connecting to `dstack`..."):
            st.session_state.client.backends.list()
            st.session_state.welcome = True
    except ClientError:
        st.warning("Can't connect to the `dstack` server")
        st.write("Make sure the `dstack` server is up:")
        st.code(
            """
                dstack server
            """,
            language="shell",
        )
        if st.button("Go to docs"):
            webbrowser.open_new_tab("https://dstack.ai/docs")
        st.stop()

st.info("Select **Fine-tuning** or **Inference**")
