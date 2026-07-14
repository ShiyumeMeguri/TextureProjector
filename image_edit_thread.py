"""
Background thread for AI image editing (Image Editor workflow).

The API call runs off the main thread; loading the result into Blender
and updating UI state is dispatched back through the shared main-thread
queue in threading_utils.
"""

import os
import shutil
import threading
from datetime import datetime
from typing import Optional

import bpy

from .threading_utils import (execute_in_main_thread, build_output_filepath,
                              show_error_popup)


class ImageEditThread(threading.Thread):
    """One background edit job."""

    def __init__(self, image_path: str, edit_prompt: str,
                 mask_path: Optional[str], reference_path: Optional[str],
                 api_key: str, original_image_name: str, temp_dir: str,
                 resolution: str = 'AUTO', original_size: tuple = (1024, 1024)):
        super().__init__(daemon=True)
        self.image_path = image_path
        self.edit_prompt = edit_prompt
        self.mask_path = mask_path
        self.reference_path = reference_path
        self.api_key = api_key
        self.original_image_name = original_image_name
        self.temp_dir = temp_dir
        self.resolution = resolution
        self.original_size = original_size

        self.result_image_data = None
        self.error_message = None

    def _resolve_resolution(self):
        if self.resolution in {'1024', '2048', '4096'}:
            size = int(self.resolution)
            return size, size
        # AUTO: keep the tier of the input image.
        orig_w, orig_h = self.original_size
        max_dim = max(orig_w, orig_h, 1)
        if max_dim > 2048:
            return 4096, 4096
        if max_dim > 1024:
            return 2048, 2048
        return 1024, 1024

    def run(self):
        try:
            self._update_status("Sending to AI...")

            from . import gemini_api
            api_client = gemini_api.GeminiAPI(self.api_key)

            width, height = self._resolve_resolution()
            print(f"[GEMINI] Edit request: prompt='{self.edit_prompt[:80]}' "
                  f"mask={bool(self.mask_path)} ref={bool(self.reference_path)} "
                  f"target={width}x{height}")

            image_data, mime_type = api_client.edit_image(
                image_path=self.image_path,
                edit_prompt=self.edit_prompt,
                mask_path=self.mask_path,
                reference_image_path=self.reference_path,
                width=width,
                height=height,
            )
            print(f"[GEMINI] Edit complete: {len(image_data)} bytes ({mime_type})")

            self.result_image_data = image_data
            self._update_status("Loading result...")
            self._load_result_in_main_thread()
            self._add_to_history()
            self._update_status("Edit complete")

        except Exception as e:
            self.error_message = str(e)
            print(f"[GEMINI] Edit thread error: {self.error_message}")
            import traceback
            traceback.print_exc()
            self._update_status(f"Error: {self.error_message[:80]}")
            show_error_popup(self.error_message)
        finally:
            self._cleanup_temp_files()

            def reset_flag():
                props = bpy.context.window_manager.nano_banana_editor
                props.is_editing = False
            execute_in_main_thread(reset_flag)

    # -- main-thread callbacks ------------------------------------------------

    def _update_status(self, message: str):
        def update():
            props = bpy.context.window_manager.nano_banana_editor
            props.status_text = message
        execute_in_main_thread(update)

    def _load_result_in_main_thread(self):
        if not self.result_image_data:
            return

        def load_image():
            try:
                result_path = build_output_filepath(
                    f"{self.original_image_name}_edit")
                with open(result_path, 'wb') as f:
                    f.write(self.result_image_data)

                timestamp = datetime.now().strftime("%H%M%S")
                new_name = f"{self.original_image_name}_edit_{timestamp}"

                new_image = bpy.data.images.load(result_path, check_existing=False)
                new_image.name = new_name
                new_image.filepath_raw = result_path
                if hasattr(new_image, 'colorspace_settings'):
                    new_image.colorspace_settings.name = 'sRGB'

                # Show the result in every open Image Editor.
                shown = False
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            for space in area.spaces:
                                if space.type == 'IMAGE_EDITOR':
                                    space.image = new_image
                                    space.mode = 'VIEW'
                                    shown = True
                            area.tag_redraw()
                if not shown:
                    print("[GEMINI] No Image Editor open to display the result")
                print(f"[GEMINI] Result loaded as: {new_name}")
            except Exception as e:
                print(f"[GEMINI] Error loading edit result: {e}")
                import traceback
                traceback.print_exc()

        execute_in_main_thread(load_image)

    def _add_to_history(self):
        def add_history():
            try:
                props = bpy.context.window_manager.nano_banana_editor
                item = props.edit_history.add()
                item.prompt = self.edit_prompt
                item.image_name = self.original_image_name
                item.timestamp = datetime.now().strftime("%H:%M:%S")
                item.has_mask = bool(self.mask_path)
            except Exception as e:
                print(f"[GEMINI] Error adding edit history: {e}")
        execute_in_main_thread(add_history)

    def _cleanup_temp_files(self):
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                # Safety: only remove our own working directories.
                if os.path.basename(self.temp_dir).startswith("nano_banana_edit_"):
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
        except OSError as e:
            print(f"[GEMINI] Temp cleanup warning: {e}")
