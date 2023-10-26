import time
from typing import Optional
import requests
import streamlit as st
from dstack.api import Client, GPU, ClientError, Resources, Task, PortUsedError

from utils import get_model, get_total_memory

st.title("Inference")
st.caption(
    "This app allows to deploy any LLM to the cloud and access it for inference."
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
    try:
        with st.spinner("Connecting to `dstack`..."):
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


def get_configuration(model_id: str, quantize: Optional[str]):
    args = ""
    if quantize:
        args = f"--quantize {quantize}"
    else:
        args = "--dtype float16"
    return Task(
        image="ghcr.io/huggingface/text-generation-inference:latest",
        env={"MODEL_ID": model_id},
        commands=[
            f"text-generation-launcher --trust-remote-code {args}",
        ],
        ports=["8080:80"],
    )



with st.sidebar:
    model_id = st.text_input(
        "Model name",
        st.session_state.model_id,
        placeholder="A repo name on Hugging Face",
        disabled=st.session_state.deploying or st.session_state.deployed,
    )

    def example_on_click(model_id):
        st.session_state.model_id = model_id

    st.caption("Examples:")
    st.button("TheBloke/Llama-2-70B-chat-AWQ", on_click=example_on_click, args=("TheBloke/Llama-2-70B-chat-AWQ",))
    st.button("TheBloke/CodeLlama-34B-AWQ", on_click=example_on_click, args=("TheBloke/CodeLlama-34B-AWQ",))

    with st.expander("Settings"):
        backend_options = ["No preference"]
        for backend in st.session_state.client.backends.list():
            backend_options.append(backend.name)
        if model_id:
            model = get_model(model_id, st.session_state.hf_token or None)
            if hasattr(model.config, "quantization_config"):
                _q = model.config.quantization_config["quant_method"]
                if _q in ["gptq", "awq"]:
                    quant_method = _q
            else:
                quant_method = None
            if not quant_method and "GPTQ" in model_id:
                quant_method = "gptq"
            elif not quant_method and "AWQ" in model_id:
                quant_method = "awq"
            # Minium size of 24GB is used as a workaround to make sure the GPU is of at least Ampere architecture (required by Flash Attention 2)

        quantize_labels = ["No quantization", "AWQ", "GPTQ"]
        quantize_options = [None, "awq", "gptq"]
        quantize_label = st.selectbox("Quantization", quantize_labels if model_id else ["Select a model"], 
                                            index=quantize_options.index(quant_method) if model_id else 0,
                                            disabled=not model_id or st.session_state.deploying or st.session_state.deployed)
        if model_id:
            quantize = quantize_options[quantize_labels.index(quantize_label)]
            estimated_gpu_memory = get_total_memory(model, dtype="int4" if quant_method else "float16", dtype_minimum_size=25769803776)
        else: 
            estimated_gpu_memory = None

        gpu_memory = st.text_input("vRAM", estimated_gpu_memory, placeholder="Select a model", disabled=st.session_state.deploying or st.session_state.deployed)
    
        # st.session_state.hf_token = settings.text_input("Hugging Face API token", st.session_state.hf_token, type="password")
        try:
            run_backend = st.session_state.run.backend
        except:
            run_backend = None
        backend_option = st.selectbox(
            "Cloud",
            backend_options,
            disabled=st.session_state.deploying or st.session_state.deployed,
            index=backend_options.index(run_backend)
            if st.session_state.run
            else 0
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
    status = st.status(
        "Deploying the LLM..."
        if not st.session_state.run
        else "Attaching to the LLM...",
        expanded=True,
    )
    if not st.session_state.run:
        status.write("Submitting...")
        try:
            run = st.session_state.client.runs.submit(
                configuration=get_configuration(model_id, quantize=quantize),
                run_name=run_name,
                resources=Resources(gpu=GPU(memory=gpu_memory)),
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
                    st.session_state.model_id = r.json()["model_id"]
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


if not st.session_state.deployed:
    st.info("The LLM is down")
else:
    st.info("The LLM is up!")
    st.markdown(
        "Fee to access the LLM at at [`http://127.0.0.1:8080`](http://127.0.0.1:8080/docs)"
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

