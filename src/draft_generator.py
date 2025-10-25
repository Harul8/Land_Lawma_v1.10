# src/draft_generator.py
# ------------------------------------------------------------
# Draft generation module optimized for Mistral 7B on 8GB GPU
# ------------------------------------------------------------

import time

class DraftGenerator:
    def __init__(self, llm_interface):
        """
        Handles legal draft/opinion generation using the loaded LLM.

        Args:
            llm_interface (LLMInterface): Interface to the local Mistral model.
        """
        self.llm = llm_interface

    def _normalize_context(self, context):
        """
        Safely normalize any type of context (list, tuple, dict, str) into clean text.
        """
        if isinstance(context, (list, tuple)):
            # Join list items into readable chunks separated by dividers
            return "\n\n---\n\n".join([str(item).strip() for item in context if item])
        elif isinstance(context, dict):
            # Convert dict to a readable key:value format
            return "\n".join([f"{k}: {v}" for k, v in context.items()])
        elif isinstance(context, str):
            return context.strip()
        else:
            # Fallback for any unexpected data type
            return str(context).strip()

    def _build_prompt(self, facts, retrieved_context):
        """
        Compose a clear, structured prompt for the LLM.

        Args:
            facts (str): User-provided facts or case background.
            retrieved_context (Union[str, list, dict]): Extracted context from relevant PDFs.

        Returns:
            str: Combined, formatted prompt ready for generation.
        """
        # ✅ Normalize all inputs first
        facts_clean = str(facts).strip()
        context_clean = self._normalize_context(retrieved_context)

        prompt_text = f"""
You are a senior legal associate specializing in Indian law.

Your task is to draft a professional, logically structured legal opinion
based on the provided case facts and supporting legal materials.

---
CASE FACTS:
{facts_clean}

---
REFERENCE MATERIAL:
{context_clean}

---
DRAFTING INSTRUCTIONS:
1. Write a concise, well-reasoned legal draft (200–250 words).
2. Use formal legal language and structured paragraphs.
3. Base reasoning on statutory provisions or case law when relevant.
4. Avoid repetition, speculation, or unnecessary filler.
---

Now write the final legal draft below:
"""
        # ✅ Always return as a clean string
        return str(prompt_text).strip()

    def generate(
        self,
        facts,
        retrieved_context,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        do_sample: bool = False,
    ):
        """
        Generate the legal draft from facts + context.

        Args:
            facts (str): Case facts from the user.
            retrieved_context (Union[str, list, dict]): Contextual information from documents.
            max_new_tokens (int): Maximum tokens for generation.
            temperature (float): Lower = more deterministic and precise.
            do_sample (bool): False = deterministic, True = more creative.

        Returns:
            str: Generated draft text.
        """
        # ✅ Build the final prompt safely
        prompt = self._build_prompt(facts, retrieved_context)

        print("\n🧩 Generating draft using Mistral model... please wait.\n")
        start = time.time()

        try:
            # Call the LLM for generation
            output_text = self.llm.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return "⚠️ Draft generation failed. Please check model or GPU status."

        end = time.time()
        print(f"✅ Draft generated in {end - start:.2f}s\n")

        # ✅ Clean and extract draft body
        output_text = str(output_text).strip()
        draft = output_text.split("Now write the final legal draft below:")[-1].strip()

        return draft or output_text
