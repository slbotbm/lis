# Not working
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_exaone_4_2b(prompt: str, reasoning: bool = True) -> str:
    """
    Generates text using the LGAI-EXAONE/EXAONE-4.0-1.2B model.

    Args:
        system_prompt: The system prompt to guide the model's behavior.
        prompt: The user's prompt for the model.
        reasoning: A boolean to enable or disable the model's thinking process.

    Returns:
        The final generated content as a string, excluding any thinking process.
    """
    model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="bfloat16", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        messages = []
        messages.append({"role": "user", "content": prompt})

        # For the EXAONE model, reasoning is activated by passing `enable_thinking=True`. [1]
        # This prompts the model to generate a <think> block with its reasoning process. [6]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=reasoning,
        )

        if reasoning:
            generation_config = {
                "max_new_tokens": 32768,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
            }
        else:
            generation_config = {
                "max_new_tokens": 32768,
                "do_sample": False,
            }

        output_ids = model.generate(
            input_ids.to(model.device),
            **generation_config,
        )

        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result
