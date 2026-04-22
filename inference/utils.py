"""Inference utility helpers."""

import re


def postprocess_caption(caption: str) -> str:
    """Remove start/end tokens and clean up the generated caption."""
    caption = re.sub(r"\bstartseq\b", "", caption)
    caption = re.sub(r"\bendseq\b", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    if caption and not caption.endswith("."):
        caption += "."
    return caption
