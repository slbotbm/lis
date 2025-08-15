import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_phi4_mini(system_prompt: str, prompt: str) -> str:
    """
    Generates text using the microsoft/Phi-4-mini-instruct model.

    Args:
        system_prompt: The system message to set the context for the model.
        prompt: The user's prompt to the model.

    Returns:
        The generated text from the model.
    """
    model_path = "microsoft/Phi-4-mini-instruct"
    model = None
    tokenizer = None
    result = ""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }

        with torch.inference_mode():
            outputs = model.generate(inputs, **generation_args)

        input_length = inputs.shape[1]
        generated_tokens = outputs[0][input_length:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result
