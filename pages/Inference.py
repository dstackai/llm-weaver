import time
from typing import Optional

import requests
import streamlit as st
from dstack.api import GPU, Client, ClientError, PortUsedError, Resources, Task

from utils import get_model, get_total_memory

st.title("Inference")
st.caption(
    "This app allows to deploy any LLM via user interface and access it for inference."
)

run_name = "llm-weaver-inference"

if "deploying" not in st.session_state:
    st.session_state.deploying = False
    st.session_state.deployed = False
    st.session_state.client = Client.from_config()
    st.session_state.run = None
    st.session_state.error = None
    st.session_state.model_id = None
    st.session_state.hf_token = None
    st.session_state.estimated_gpu_memory = None
    st.session_state.quantize = None
    try:
        with st.spinner("Connecting to `dstack`..."):
            backend_options = ["No preference"]
            for backend in st.session_state.client.backends.list():
                backend_options.append(backend.name)
            st.session_state.backend_options = backend_options

            run = st.session_state.client.runs.get(run_name)
            if run and not run.status.is_finished():
                st.session_state.run = run
                st.session_state.deploying = True
    except ClientError:
        st.info("Can't connect to `dstack`")
        st.text("Make sure the `dstack` server is up:")
        st.code(
            """
                dstack server
            """,
            language="shell",
        )
        st.stop()


def deploy_on_click():
    st.session_state.deploying = True
    st.session_state.error = None


def undeploy_on_click():
    st.session_state.run.stop()
    st.session_state.deploying = False
    st.session_state.deployed = False
    st.session_state.run = None


def get_configuration():
    args = ""
    if st.session_state.quantize == "GPTQ":
        args = f"--quantize gptq"
    elif st.session_state.quantize == "AWQ":
        args = f"--quantize awq"
    else:
        args = "--dtype float16"
    return Task(
        image="ghcr.io/huggingface/text-generation-inference:latest",
        env={"MODEL_ID": st.session_state.model_id},
        commands=[
            f"text-generation-launcher --trust-remote-code {args}",
        ],
        ports=["8080:80"],
    )


def get_resources():
    return Resources(gpu=GPU(memory=st.session_state.estimated_gpu_memory))


e = st.empty()
placeholder = e.container()

c1, c2 = placeholder.columns(2)

quantize_labels = ["No quantization", "AWQ", "GPTQ"]
quantize_options = [None, "awq", "gptq"]

if not st.session_state.deploying and not st.session_state.deployed:

    def model_id_on_change():
        if st.session_state.model_id:
            model = get_model(st.session_state.model_id, st.session_state.hf_token)

            quantize = "No quantization"
            if "GPTQ" in st.session_state.model_id:
                quantize = "GPTQ"
            elif "AWQ" in st.session_state.model_id:
                quantize = "AWQ"
            st.session_state.quantize = quantize

            # Minium size of 24GB is used as a workaround to make sure the GPU is of at least Ampere architecture (required by Flash Attention 2)
            st.session_state.estimated_gpu_memory = get_total_memory(
                model,
                dtype="int4" if quantize != "No quantization" else "float16",
                dtype_minimum_size=25769803776,
            )
        else:
            st.session_state.quantize = None
            st.session_state.estimated_gpu_memory = None

    model_id = c1.text_input(
        "Model name",
        placeholder="A repo name on Hugging Face",
        disabled=st.session_state.deploying or st.session_state.deployed,
        on_change=model_id_on_change,
        key="model_id",
    )

    def example_on_click(model_id):
        st.session_state.model_id = model_id
        model_id_on_change()

    c2.caption("Examples:")
    c2.button(
        "TheBloke/Llama-2-70B-chat-GPTQ",
        on_click=example_on_click,
        args=("TheBloke/Llama-2-70B-chat-GPTQ",),
    )
    c2.button(
        "TheBloke/CodeLlama-34B-GPTQ",
        on_click=example_on_click,
        args=("TheBloke/CodeLlama-34B-GPTQ",),
    )

with st.sidebar:
    st.selectbox(
        "Quantization",
        quantize_labels if st.session_state.model_id else ["Select a model"],
        disabled=not st.session_state.model_id
        or st.session_state.deploying
        or st.session_state.deployed,
        key="quantize",
    )

    st.text_input(
        "vRAM",
        placeholder="Select a model",
        disabled=st.session_state.deploying or st.session_state.deployed,
        key="estimated_gpu_memory",
    )

    # st.session_state.hf_token = settings.text_input("Hugging Face API token", st.session_state.hf_token, type="password")

    try:
        run_backend = st.session_state.run.backend
    except:
        run_backend = None
    backend_option = st.selectbox(
        "Cloud",
        st.session_state.backend_options,
        disabled=st.session_state.deploying or st.session_state.deployed,
        index=st.session_state.backend_options.index(run_backend) if run_backend else 0,
    )

if not st.session_state.deploying and not st.session_state.deployed:
    st.button(
        "Deploy",
        on_click=deploy_on_click,
        type="primary",
        disabled=not model_id,
    )


if st.session_state.error:
    with st.sidebar:
        st.error(st.session_state.error)


if st.session_state.deploying:
    del e
    status = st.status(
        "Deploying the LLM..."
        if not st.session_state.run
        else "Attaching to the LLM...",
        expanded=True,
    )
    if not st.session_state.run:
        try:
            run = st.session_state.client.runs.submit(
                configuration=get_configuration(),
                run_name=run_name,
                resources=get_resources(),
                backends=None
                if backend_option == "No preference"
                else [backend_option],
            )
            st.session_state.run = run
            status.write("Attaching...")
        except Exception as e:
            if hasattr(e, "message"):
                st.session_state.error = e.message
            else:
                raise e
            st.session_state.deploying = False
            st.rerun()
    if not st.session_state.error:
        placeholder = st.empty()
        placeholder.button(
            "Cancel",
            type="primary",
            key="cancel_starting",
            on_click=undeploy_on_click,
        )
        try:
            st.session_state.run.attach()
        except PortUsedError:
            pass
        while True:
            time.sleep(0.5)
            try:
                r = requests.get("http://localhost:8080/info")
                if r.status_code == 200:
                    st.session_state.deployed = True
                    break
                else:
                    st.session_state.run.refresh()
                    if st.session_state.run.status.is_finished():
                        st.session_state.error = "Failed or interrupted"
                        break
            except Exception as e:
                pass
    st.session_state.deploying = False
    st.rerun()


if st.session_state.deployed:
    st.info("The LLM is up!")
    st.markdown(
        "Feel free to access the LLM at at [`http://127.0.0.1:8080`](http://127.0.0.1:8080/docs)"
    )
    st.code(
        """curl 127.0.0.1:8080 \\
    -X POST \\
    -d '{"inputs":"<prompt>","parameters":{"max_new_tokens":20}}' \\
    -H 'Content-Type: application/json'
            """,
        language="shell",
    )
    st.button(
        "Undeploy",
        type="primary",
        key="stop",
        on_click=undeploy_on_click,
    )
