import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_llama3_1(system_prompt: str, user_prompt: str) -> str:
    """
    Generates text using the meta-llama/Meta-Llama-3.1-8B-Instruct model
    by directly using the AutoModel and AutoTokenizer classes.

    Args:
        system_prompt: The system prompt to guide the model's behavior.
        user_prompt: The user's prompt for the model.

    Returns:
        The generated text as a string.
    """
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = None
    tokenizer = None

    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create the message list with system and user prompts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply the chat template to format the input
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Define the terminators to stop generation
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # Generate the model's response
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # Extract and decode the generated text
        response = outputs[0][input_ids.shape[-1] :]
        generated_text = tokenizer.decode(response, skip_special_tokens=True)

    finally:
        # Clean up to free memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return generated_text


if __name__ == "__main__":
    generate_llama3_1("", "")
