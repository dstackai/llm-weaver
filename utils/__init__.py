from typing import Optional
from accelerate.commands.estimate import create_empty_model
from accelerate.utils import calculate_maximum_sizes



def get_model(model_id: str, hf_token: Optional[str] = None):
    
    try:
        model = create_empty_model(
            model_id, library_name=None, trust_remote_code=True, access_token=hf_token
        )
    except ImportError:
        model = create_empty_model(
            model_id, library_name=None, trust_remote_code=False, access_token=hf_token
        )
    return model


def _convert_bytes(size) -> str:
    for x in ["bytes", "KB", "MB"]:
        if size < 1024.0:
            return f"{round(size)}{x}"
        size /= 1024.0

    return f"{round(size)}GB"

def get_total_memory(model, dtype: str = "float32", dtype_minimum_size: Optional[int] = None) -> str:
    modifiers = {"float32": 1, "float16": 2, "bfloat16": 2, "int8": 4, "int4": 8}
    modifier = modifiers[dtype]

    dtype_total_size, _ = calculate_maximum_sizes(model)
    dtype_total_size /= modifier
    
    if dtype_total_size < dtype_minimum_size:
        dtype_total_size = dtype_minimum_size

    return _convert_bytes(dtype_total_size)
