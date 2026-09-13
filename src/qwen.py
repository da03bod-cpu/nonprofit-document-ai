import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from prompts import SYSTEM_PROMPT, build_user_prompt


class QwenExtractor:
    def __init__(self):
        self.model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B")
        self.max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "4096"))
        self.max_input_chars = int(os.getenv("QWEN_MAX_INPUT_CHARS", "100000"))
        load_4bit = os.getenv("QWEN_LOAD_IN_4BIT", "true").lower() == "true"

        print(f"Loading Qwen3: {self.model_name}")
        print(f"4-bit loading: {load_4bit}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        self.model.eval()
        print("Qwen3 loaded.")

    def generate(self, text: str):
        if not text or not text.strip():
            raise ValueError("No document text available for Qwen3")

        # Prevent accidentally sending an enormous report beyond the
        # configured context budget. Later we can replace this with
        # semantic chunking/merging for very large reports.
        text = text[:self.max_input_chars]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        # Remove accidental markdown fences if the model emits them.
        if answer.startswith("```"):
            answer = answer.replace("```json", "", 1).replace("```", "", 1).strip()

        try:
            return json.loads(answer)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Qwen3 did not return valid JSON. Raw output: {answer[:5000]}"
            ) from exc
