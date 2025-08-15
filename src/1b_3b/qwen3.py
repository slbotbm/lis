import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_qwen3_1_7b(
    system_prompt: str, prompt: str, reasoning: bool = False
) -> tuple[str, str]:
    """
    Generates text using the Qwen/Qwen3-8B model with an option for a system prompt
    and to enable or disable reasoning.

    Args:
        system_prompt: The system prompt to guide the model's overall behavior.
        user_prompt: The user's specific prompt for the model.
        reasoning: A boolean to enable or disable the model's thinking process.

    Returns:
        A tuple containing the thinking content and the final generated content.
    """
    model_name = "Qwen/Qwen3-1.7B"
    try:
        # Load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=reasoning,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Conduct text completion
        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Parsing thinking content
        try:
            # Find the last occurrence of the </think> token (ID 151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            # If the token isn't found, set index to 0
            index = 0

        result = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip(
            "\n"
        )

    finally:
        # Clean up memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return result
