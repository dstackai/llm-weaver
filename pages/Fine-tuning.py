import json
import time

import streamlit as st
from code_editor import code_editor
from dstack.api import GPU, Client, ClientError, Resources
from dstack.api.huggingface import SFTFineTuningTask

from utils import get_model, get_total_memory, validate

st.title("Fine-tuning")
st.caption("This app allows to fine-tune any LLM via user interface.")

run_name = "llm-weaver-fine-tuning"

if "fine_tuning" not in st.session_state:
    st.session_state.fine_tuning = False
    st.session_state.fine_tuned = False
    st.session_state.client = Client.from_config()
    st.session_state.run = None
    st.session_state.error = None
    st.session_state.model_id = None
    st.session_state.dataset_id = None
    st.session_state.new_model_id = None
    st.session_state.hf_token = None
    st.session_state.hf_token_valid = None
    st.session_state.wandb_api_key = None
    st.session_state.parameters = json.dumps(
        {
            "num_train_epochs": 2,
            "max_seq_length": 1024,
            "per_device_train_batch_size": 2,
        },
        indent=2,
    )
    try:
        with st.spinner("Connecting to `dstack`..."):
            backend_options = ["No preference"]
            for backend in st.session_state.client.backends.list():
                backend_options.append(backend.name)
            st.session_state.backend_options = backend_options

            run = st.session_state.client.runs.get(run_name)
            if run and not run.status.is_finished():
                st.session_state.run = run
                st.session_state.fine_tuning = True
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


def get_resources():
    return Resources(gpu=GPU(memory=st.session_state.estimated_gpu_memory))


def fine_tune_on_click():
    st.session_state.fine_tuning = True
    st.session_state.error = None


def cancel_on_click():
    st.session_state.run.stop()
    st.session_state.fine_tuning = False
    st.session_state.fine_tuned = False
    st.session_state.run = None


def hf_token_on_change():
    if st.session_state.hf_token:
        try:
            validate(st.session_state.hf_token)
            st.session_state.hf_token_valid = True
        except Exception:
            st.sidebar.error("API token is invalid")
            st.session_state.hf_token_valid = False
    else:
        st.session_state.hf_token_valid = False


with st.sidebar:
    hf_token = st.text_input(
        "Hugging Face API token",
        st.session_state.hf_token,
        type="password",
        key="hf_token",
        placeholder="Required",
        on_change=hf_token_on_change,
        disabled=st.session_state.fine_tuning,
    )

    try:
        run_backend = st.session_state.run.backend
    except:
        run_backend = None
    backend_option = st.selectbox(
        "Cloud",
        st.session_state.backend_options,
        disabled=st.session_state.fine_tuning,
        index=st.session_state.backend_options.index(run_backend)
        if st.session_state.run
        else 0,
    )

    st.text_input(
        "GPU memory",
        placeholder="Select a model",
        disabled=st.session_state.fine_tuning,
        key="estimated_gpu_memory",
    )

    report_to = st.selectbox(
        "Report to",
        ["TensorBoard", "Weights & Biases"],
        key="report_to",
        disabled=st.session_state.fine_tuning,
    )

    if report_to == "Weights & Biases":
        st.text_input(
            "WANDB_API_KEY",
            placeholder="Required",
            key="wandb_api_key",
            type="password",
        )

e = st.empty()
placeholder = e.container()

if not st.session_state.fine_tuning:
    c1, c2 = placeholder.columns(2)

    def model_id_on_change():
        # if st.session_state.model_id:
        #     model = get_model(st.session_state.model_id, st.session_state.hf_token)
        #     st.session_state.estimated_gpu_memory = get_total_memory(
        #         model,
        #         dtype="int4",
        #     )
        # else:
        #     st.session_state.estimated_gpu_memory = None
        pass

    model_id = c1.text_input(
        "Model name",
        st.session_state.model_id,
        placeholder="A repo name on Hugging Face",
        disabled=st.session_state.fine_tuning,
        on_change=model_id_on_change,
        key="model_id",
    )

    def example_on_click(model_id):
        st.session_state.model_id = model_id
        model_id_on_change()

    c1.caption("Example:")
    c1.button(
        "NousResearch/Llama-2-13b-hf",
        on_click=example_on_click,
        args=("NousResearch/Llama-2-13b-hf",),
    )

    dataset_id = c2.text_input(
        "Dataset name",
        st.session_state.dataset_id,
        placeholder="A repo name on Hugging Face",
        disabled=st.session_state.fine_tuning,
    )

    def example_2_on_click(dataset_id):
        st.session_state.dataset_id = dataset_id

    c2.caption("Example:")
    c2.button(
        "peterschmidt85/samsum",
        on_click=example_2_on_click,
        args=("peterschmidt85/samsum",),
    )

    fine_tuning_type = c1.selectbox(
        "Fine-tuning type", ["SFT"], key="sft", disabled=True
    )

    new_model_prefix = (
        model_id.split("/")[1] if model_id and model_id.index("/") else model_id
    )
    new_model_suffix = (
        dataset_id.split("/")[1] if dataset_id and dataset_id.index("/") else dataset_id
    )
    new_model_id = c2.text_input(
        "New model name",
        f"{new_model_prefix}-{new_model_suffix}"
        if new_model_prefix and new_model_suffix
        else "",
        key="new_model_id",
    )

    with placeholder.expander("Parameters"):
        code_editor(st.session_state.parameters, lang="json", key="parameters")
        st.caption(
            "Learn more <a href='https://dstack.ai/docs/reference/api/python/#dstack.api.huggingface.SFTFineTuningTask'>here</a>.",
            unsafe_allow_html=True,
        )

    placeholder.button(
        "Fine-tune",
        on_click=fine_tune_on_click,
        type="primary",
        disabled=not st.session_state.model_id
        or not st.session_state.dataset_id
        or not st.session_state.hf_token_valid
        or not st.session_state.new_model_id
        or not st.session_state.estimated_gpu_memory
        or (report_to == "Weights & Biases" and not st.session_state.wandb_api_key),
    )

if st.session_state.fine_tuning:
    status = st.status("Fine-tuning...", expanded=False)
    with status:
        log_area = st.empty()
    status.update(state="running")
    if not st.session_state.run:
        params = json.loads(st.session_state.parameters)
        env = {
            "HUGGING_FACE_HUB_TOKEN": st.session_state.hf_token,
            **params,
        }
        if report_to == "Weights & Biases":
            env["WANDB_API_KEY"] = st.session_state.wandb_api_key
        configuration = SFTFineTuningTask(
            model_name=st.session_state.model_id,
            dataset_name=st.session_state.dataset_id,
            new_model_name=st.session_state.new_model_id,
            env=env,
            report_to="wandb" if report_to == "Weights & Biases" else None,
        )
        resources = get_resources()
        try:
            run = st.session_state.client.runs.submit(
                configuration=configuration,
                run_name=run_name,
                resources=resources,
                backends=None
                if backend_option == "No preference"
                else [backend_option],
            )
            st.session_state.run = run
        except Exception as e:
            if hasattr(e, "message"):
                st.session_state.error = e.message
            else:
                raise e
            st.session_state.fine_tuning = False
            st.rerun()
    if not st.session_state.error:
        placeholder = st.empty()
        placeholder.button(
            "Cancel",
            type="primary",
            key="cancel_starting",
            on_click=cancel_on_click,
        )
        st.session_state.run.attach()
        logs = ""
        st.markdown(
            "<style>[data-testid=stMarkdownContainer] pre { max-height: 400px; }</style>",
            unsafe_allow_html=True,
        )
        for log in run.logs():
            logs += f"{str(log)}\n"
            log_area.markdown(f"```{logs}```")
