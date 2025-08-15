# transformers==4.55.0
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_afm_4_5b(system_prompt: str, prompt: str) -> str:
    """
    Generates text using the arcee-ai/AFM-4.5B model.

    Args:
        prompt: The user's prompt for the model.
        system_prompt: An optional system prompt to guide the model.

    Returns:
        The generated text from the model.
    """
    model_id = "arcee-ai/AFM-4.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.5,
                top_k=50,
                top_p=0.95,
            )

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
