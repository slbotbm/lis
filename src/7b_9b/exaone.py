import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_exaone(system_prompt: str, user_prompt: str) -> str:
    """
    Generates text using the LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct model.

    Args:
        system_prompt: The system prompt to guide the model's behavior.
        user_prompt: The user's prompt for the model.

    Returns:
        The generated text as a string.
    """
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    model = None
    tokenizer = None
    try:
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create the message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply the chat template to format the input
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Generate the response
        output = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,  # Increased token limit for more detailed answers
            do_sample=False,
        )

        # Decode only the newly generated tokens
        response_ids = output[0][len(input_ids[0]) :]
        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    finally:
        # Clean up to free memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return generated_text
