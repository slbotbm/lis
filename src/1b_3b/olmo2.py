import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_olmo_1b(system_prompt: str, prompt: str) -> str:
    """
    Generates a response from the OLMo-2-0425-1B-Instruct model.

    Args:
        system_prompt: The system prompt to guide the model's behavior.
        user_prompt: The user's prompt to the model.

    Returns:
        The model's generated response as a string.
    """
    model_name = "allenai/OLMo-2-0425-1B-Instruct"
    model = None
    tokenizer = None
    result = ""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # The chat template is embedded within the tokenizer. [1]
        # We create a list of dictionaries representing the conversation.
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs.to(model.device),
                max_new_tokens=32768,
                do_sample=True,
                top_p=0.95,
            )

        input_length = inputs["input_ids"].shape[-1]
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_length:])
        result = decoded_outputs[0].strip().replace("<|endoftext|>", "")

    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result
