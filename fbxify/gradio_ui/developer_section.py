"""
Gradio UI components for developer section.

This module provides UI components for developer/debug options including
visualization, debug save/load, and reexport functionality.
"""
import gradio as gr
from typing import Dict, Any, List, Callable, Tuple
from datetime import datetime
from fbxify.i18n import Translator


def create_developer_section(translator: Translator, 
                             get_timestamps_fn: Callable[[], List[int]]) -> Dict[str, Any]:
    """
    Create the developer section UI components.
    
    Args:
        translator: Translator instance for i18n
        get_timestamps_fn: Function to get list of saved timestamps
        
    Returns:
        Dictionary of component names to Gradio components
    """
    components = {}
    
    with gr.Accordion(translator.t("ui.developer_options"), open=False):
        components['create_visualization'] = gr.Checkbox(
            label=translator.t("ui.create_visualization"),
            value=False
        )
        
        gr.Markdown("---")
        gr.Markdown("### Debug: Save & Re-export")
        
        components['debug_save_results'] = gr.Checkbox(
            label=translator.t("ui.debug_save_results"),
            value=False
        )
        
        def get_timestamp_choices():
            timestamps = get_timestamps_fn()
            if not timestamps:
                return [translator.t("ui.debug_no_saved_results")]
            # Convert timestamps to readable format
            choices = []
            for ts in reversed(timestamps):  # Show newest first
                dt = datetime.fromtimestamp(ts)
                label = f"{dt.strftime('%Y-%m-%d %H:%M:%S')} ({ts})"
                choices.append((label, str(ts)))
            return choices
        
        components['debug_saved_timestamps'] = gr.Dropdown(
            label=translator.t("ui.debug_saved_timestamps"),
            choices=get_timestamp_choices(),
            value=None,
            interactive=True
        )
        
        with gr.Row():
            components['debug_refresh_btn'] = gr.Button("🔄 Refresh List", size="sm")
            components['debug_clear_btn'] = gr.Button(translator.t("ui.debug_clear_btn"), size="sm", variant="stop")
        
        components['debug_reexport_btn'] = gr.Button(
            translator.t("ui.debug_reexport_btn"),
            variant="secondary"
        )
    
    return components


def refresh_timestamps(get_timestamps_fn: Callable[[], List[int]], 
                      translator: Translator) -> Any:
    """
    Refresh the timestamp dropdown choices.
    
    Args:
        get_timestamps_fn: Function to get list of saved timestamps
        translator: Translator instance
        
    Returns:
        Update for debug_saved_timestamps dropdown
    """
    timestamps = get_timestamps_fn()
    if not timestamps:
        choices = [translator.t("ui.debug_no_saved_results")]
        return gr.update(choices=choices, value=None)
    
    # Convert timestamps to readable format
    choices = []
    for ts in reversed(timestamps):  # Show newest first
        dt = datetime.fromtimestamp(ts)
        label = f"{dt.strftime('%Y-%m-%d %H:%M:%S')} ({ts})"
        choices.append((label, str(ts)))
    
    return gr.update(choices=choices, value=None)


def update_developer_language(lang: str) -> Tuple[Any, ...]:
    """
    Update developer section components with new language.
    
    Args:
        lang: Language code
        
    Returns:
        Tuple of updates for all developer components
    """
    t = Translator(lang)
    return (
        gr.update(label=t.t("ui.create_visualization")),  # create_visualization
        gr.update(label=t.t("ui.debug_save_results")),  # debug_save_results
        gr.update(label=t.t("ui.debug_saved_timestamps")),  # debug_saved_timestamps
        gr.update(value=t.t("ui.debug_reexport_btn")),  # debug_reexport_btn
        gr.update(value=t.t("ui.debug_clear_btn")),  # debug_clear_btn
    )
