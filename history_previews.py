"""
Thumbnail previews for the projection gallery.

Uses Blender's native image previews when the image is in memory, and a
timer-driven background loader for on-disk files (never loads inside a
draw() callback). Ported and adapted from nano-banana-render.
"""

import os
import bpy
import bpy.utils.previews

_custom_icons = None
_load_queue = set()
_timer_registered = False


def init_previews():
    global _custom_icons
    if _custom_icons is None:
        _custom_icons = bpy.utils.previews.new()


def clear_previews():
    global _custom_icons
    if _custom_icons is not None:
        bpy.utils.previews.remove(_custom_icons)
        _custom_icons = None


def _redraw_all_areas():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def _process_queue():
    global _timer_registered
    if _custom_icons is None or not _load_queue:
        _timer_registered = False
        return None

    filepath = _load_queue.pop()
    if filepath not in _custom_icons and os.path.exists(filepath):
        try:
            _custom_icons.load(filepath, filepath, 'IMAGE')
            _redraw_all_areas()
        except Exception as e:
            print(f"[GEMINI] Failed to load preview icon: {e}")

    if _load_queue:
        return 0.1
    _timer_registered = False
    return None


def get_preview_icon_id(filepath: str = "", image_name: str = "") -> int:
    """Return an icon_id for the gallery, 0 if not (yet) available."""
    global _timer_registered

    # 1. Native preview for in-memory images (fastest path).
    if image_name and image_name in bpy.data.images:
        img = bpy.data.images[image_name]
        try:
            preview = img.preview_ensure()
            if preview:
                return preview.icon_id
        except Exception:
            pass
        if not filepath and img.filepath:
            filepath = bpy.path.abspath(img.filepath)

    if _custom_icons is None or not filepath:
        return 0

    # 2. Custom loader for on-disk files.
    if filepath in _custom_icons:
        return _custom_icons[filepath].icon_id

    if os.path.exists(filepath):
        _load_queue.add(filepath)
        if not _timer_registered:
            bpy.app.timers.register(_process_queue, first_interval=0.1)
            _timer_registered = True
    return 0
