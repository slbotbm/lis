import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_qwen3_4b(system_prompt: str, prompt: str) -> str:
    """
    Generates text using the Qwen/Qwen3-4B-Instruct-2507 model.

    Args:
        prompt: The user's prompt for the model.

    Returns:
        The final generated content as a string.
    """
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    model = None
    tokenizer = None
    result = ""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, max_new_tokens=16384)

        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        result = content

    finally:
        # Clean up to free memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result
