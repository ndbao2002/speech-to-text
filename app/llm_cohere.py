import os
import cohere
import streamlit as st

def cohere_postprocess(text: str):
    """Post-process transcript with Cohere API."""
    # Load API key from file
    api_key = st.secrets.get("COHERE_API_KEY", "")
    if not api_key:
        st.warning("Set COHERE_API_KEY in your environment to use LLM features.")
        return None
    try:
        co = cohere.Client(api_key)
        prompt = f"""You are a helpful assistant. Given this transcript, return:
1) A 3-5 sentence summary.
2) 5-8 keywords.
3) 3-5 bullet action items if relevant.

Transcript:
{text}"""
        resp = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=300,
            temperature=0.2
        )
        return resp.generations[0].text.strip()
    except Exception as e:
        st.error(f"Cohere API call failed: {e}")
        return None
