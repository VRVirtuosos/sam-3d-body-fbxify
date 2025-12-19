"""
Gradio UI components for header section.

This module provides UI components for the header section including
title, description, and language selector.
"""
import gradio as gr
from typing import Tuple, Any
from fbxify.i18n import Translator, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE


def create_header_section(translator: Translator) -> Tuple[Any, Any, Any]:
    """
    Create the header section UI components.
    
    Args:
        translator: Translator instance for i18n
        
    Returns:
        Tuple of (heading_md, description_md, lang_selector)
    """
    # Title and heading
    heading_md = gr.Markdown(f"## {translator.t('app.heading')}")
    
    # Description with features and usage
    features = translator.get("app.features", [])
    usage = translator.get("app.usage", [])
    features_text = "\n".join([f"- {f}" for f in features])
    usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
    description_text = f"### {translator.t('app.features_title')}\n{features_text}\n\n### {translator.t('app.usage_title')}\n{usage_text}"
    description_md = gr.Markdown(description_text)
    
    # Language selector dropdown
    lang_selector = gr.Dropdown(
        label="🌐 Language / 言語 / Idioma / Langue",
        choices=[("English", "en"), ("日本語", "ja"), ("Español", "es"), ("Français", "fr")],
        value=DEFAULT_LANGUAGE,
        interactive=True
    )
    
    return heading_md, description_md, lang_selector


def update_header_language(lang: str) -> Tuple[Any, Any]:
    """
    Update header components with new language.
    
    Args:
        lang: Language code
        
    Returns:
        Tuple of updates for heading and description
    """
    t = Translator(lang)
    features = t.get("app.features", [])
    usage = t.get("app.usage", [])
    features_text = "\n".join([f"- {f}" for f in features])
    usage_text = "\n".join([f"{i+1}. {u}" for i, u in enumerate(usage)])
    description_text = f"### {t.t('app.features_title')}\n{features_text}\n\n### {t.t('app.usage_title')}\n{usage_text}"
    
    return (
        gr.update(value=f"## {t.t('app.heading')}"),  # heading
        gr.update(value=description_text),  # description
    )
