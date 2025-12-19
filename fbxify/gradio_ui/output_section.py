"""
Gradio UI components for output section.

This module provides UI components for displaying output files.
"""
import gradio as gr
from typing import Any
from fbxify.i18n import Translator


def create_output_section(translator: Translator) -> Any:
    """
    Create the output section UI component.
    
    Args:
        translator: Translator instance for i18n
        
    Returns:
        Output files component
    """
    output_files = gr.File(
        label=translator.t("ui.output_files"),
        interactive=False,
        file_count="multiple"
    )
    
    return output_files


def update_output_language(lang: str) -> Any:
    """
    Update output section component with new language.
    
    Args:
        lang: Language code
        
    Returns:
        Update for output_files component
    """
    t = Translator(lang)
    return gr.update(label=t.t("ui.output_files"))
