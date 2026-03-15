"""Utility functions for featurecoding_llm_benchmark tasks."""

import re

from datasets import Dataset


def _preprocess_hellaswag_text(text):
    """Preprocess HellaSwag text - remove WikiHow artifacts."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def _get_prompt_length_arc(doc):
    """Calculate prompt length for ARC task."""
    question = doc.get("question", "")
    choices_text = doc.get("choices", {}).get("text", [])
    total_len = len(question)
    for choice in choices_text:
        total_len += len(choice)
    return total_len


def _get_prompt_length_hellaswag(doc):
    """Calculate prompt length for HellaSwag task."""
    ctx = doc.get("ctx", "")
    endings = doc.get("endings_clean", [])
    total_len = len(ctx)
    for ending in endings:
        total_len += len(ending)
    return total_len


def _get_prompt_length_truthfulqa(doc):
    """Calculate prompt length for TruthfulQA task."""
    question = doc.get("question", "")
    choices = doc.get("mc1_targets", {}).get("choices", [])
    total_len = len(question)
    for choice in choices:
        total_len += len(choice)
    return total_len


def _get_prompt_length_winogrande(doc):
    """Calculate prompt length for Winogrande task."""
    sentence = doc.get("sentence", "")
    option1 = doc.get("option1", "")
    option2 = doc.get("option2", "")
    return len(sentence) + len(option1) + len(option2)


def _get_prompt_length_gsm8k(doc):
    """Calculate prompt length for GSM8K task."""
    question = doc.get("question", "")
    return len(question)


def process_docs_arc(dataset):
    """Sort ARC documents by prompt length (descending)."""
    docs = list(dataset)
    docs.sort(key=_get_prompt_length_arc, reverse=True)
    return Dataset.from_list(docs)


def process_docs_hellaswag(dataset):
    """Preprocess HellaSwag documents (remove WikiHow artifacts) and sort by prompt length."""

    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = dict(doc)  # Copy original fields
        out_doc["ctx"] = _preprocess_hellaswag_text(doc["activity_label"] + ": " + ctx)
        out_doc["endings_clean"] = [_preprocess_hellaswag_text(ending) for ending in doc["endings"]]
        return out_doc

    docs = [_process_doc(doc) for doc in dataset]
    docs.sort(key=_get_prompt_length_hellaswag, reverse=True)
    return Dataset.from_list(docs)


def process_docs_truthfulqa(dataset):
    """Sort TruthfulQA documents by prompt length (descending)."""
    docs = list(dataset)
    docs.sort(key=_get_prompt_length_truthfulqa, reverse=True)
    return Dataset.from_list(docs)


def process_docs_winogrande(dataset):
    """Sort Winogrande documents by prompt length (descending)."""
    docs = list(dataset)
    # Sort by total length (sentence + both options)
    docs.sort(key=_get_prompt_length_winogrande, reverse=True)
    return Dataset.from_list(docs)


def process_docs_gsm8k(dataset):
    """Sort GSM8K documents by prompt length (descending)."""
    docs = list(dataset)
    docs.sort(key=_get_prompt_length_gsm8k, reverse=True)
    return Dataset.from_list(docs)
