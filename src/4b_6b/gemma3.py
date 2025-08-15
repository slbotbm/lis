# transformers==4.55.0
import gc
import torch
from transformers import Gemma3ForCausalLM, AutoTokenizer


def generate_gemma3_4b(system_prompt: str, prompt: str) -> str:
    """
    Generates text using the google/gemma-3-4b-it model for text-only tasks.

    Args:
        system_prompt: The system prompt to guide the model's behavior.
        prompt: The user's prompt for the model.

    Returns:
        The generated text from the model.
    """
    model_id = "google/gemma-3-4b-it"
    model = None
    tokenizer = None
    result = ""
    try:
        # Use Gemma3ForCausalLM for text-only generation
        model = Gemma3ForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ],
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=32768, do_sample=False)

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result


if __name__ == "main":
    generate_gemma3_4b(system_prompt="", prompt="")
