import gc
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_exaone_deep_reasoning(prompt: str) -> str:
    """
    Generates text using the LGAI-EXAONE/EXAONE-Deep-2.4B model.

    Args:
        system_prompt: The system message to set the context for the model.
        prompt: The user's prompt to the model.

    Returns:
        The generated text from the model.
    """
    model_id = "LGAI-EXAONE/EXAONE-Deep-2.4B"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs.to(model.device),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )

        input_length = inputs["input_ids"].shape[-1]
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_length:])
        result = decoded_outputs[0]
        result = re.sub(r".*?</thought>\s*", "", result, flags=re.DOTALL)

    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return result


if __name__ == "__main__":
    generate_exaone_deep_reasoning(prompt="")
