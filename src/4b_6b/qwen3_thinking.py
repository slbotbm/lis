import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_qwen3_thinking(system_prompt: str, prompt: str) -> str:
    """
    Generates text using the Qwen/Qwen3-4B-Thinking-2507 model, separating
    the thinking process from the final content.

    Args:
        prompt: The user's prompt for the model.

    Returns:
        A tuple containing two strings:
        - The thinking process of the model.
        - The final generated content.
    """
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()

        try:
            # The thinking part ends with the token ID 151668 (</think>).
            # Find the last occurrence of the </think> token.
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            # If the </think> token is not found, assume no thinking part.
            index = 0

        result = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    finally:
        # Clean up to free memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result
