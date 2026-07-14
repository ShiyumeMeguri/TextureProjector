"""
Gemini API client for TextureProjector.

Zero-dependency design: the core path speaks REST via Python's stdlib
(urllib + base64 + json), so the addon works out of the box in any Blender
without pip-installing anything. PIL / google-genai are NOT required.

Fixes vs v1:
- Default model is the free tier: gemini-3.1-flash-image-preview.
- Response modalities are selected per model family (Gemini 3.x accepts
  IMAGE-only; the 2.5 image family requires TEXT+IMAGE).
- imageConfig.imageSize is only sent to models that accept it, with a
  graceful retry cascade on HTTP 400 (drop imageSize -> drop imageConfig
  -> flip modalities) instead of hard-failing.
- Clear, actionable error messages for 401/403/429/5xx.
"""

import os
import json
import base64
import struct
from typing import Optional, Tuple, Dict, List
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT = 300

SYSTEM_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "system_prompts.json")


class GeminiAPIError(Exception):
    """Raised for any Gemini API failure, with a user-readable message."""
    pass


# ---------------------------------------------------------------------------
# Prompt management (external, user-editable system_prompts.json)
# ---------------------------------------------------------------------------

class PromptManager:
    """Loads system prompts from the editable JSON file next to the addon."""

    DEFAULT_PROMPTS = {
        "depth_with_reference": "",
        "depth_only": "",
        "color_with_reference": "",
        "color_only": "",
        "inpainting_with_reference": "",
        "inpainting_only": "",
        "edit_integration": "",
        "edit_refinement": "",
        "finalize_composite": "",
        "default_edit_prompt": "Edit this image.",
        "default_generate_prompt": "Generate image.",
    }

    _cache = None
    _cache_mtime = 0.0

    @classmethod
    def load_prompts(cls) -> Dict[str, str]:
        """Load prompts, cached by file mtime so edits apply without restart."""
        try:
            mtime = os.path.getmtime(SYSTEM_PROMPTS_FILE)
        except OSError:
            return cls.DEFAULT_PROMPTS.copy()

        if cls._cache is not None and mtime == cls._cache_mtime:
            return cls._cache

        prompts = cls.DEFAULT_PROMPTS.copy()
        try:
            with open(SYSTEM_PROMPTS_FILE, "r", encoding="utf-8") as f:
                prompts.update(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[GEMINI] Failed to load system prompts: {e}")
        cls._cache = prompts
        cls._cache_mtime = mtime
        return prompts


# ---------------------------------------------------------------------------
# Aspect ratio / resolution mapping
# ---------------------------------------------------------------------------

_ASPECT_RATIOS = {
    "1:1": 1.0,
    "2:3": 2 / 3, "3:2": 3 / 2,
    "3:4": 3 / 4, "4:3": 4 / 3,
    "4:5": 4 / 5, "5:4": 5 / 4,
    "9:16": 9 / 16, "16:9": 16 / 9,
    "21:9": 21 / 9,
}


def closest_aspect_ratio(width: int, height: int) -> str:
    ratio = (width / height) if height > 0 else 1.0
    return min(_ASPECT_RATIOS.items(), key=lambda kv: abs(kv[1] - ratio))[0]


def resolution_tier(width: int, height: int) -> str:
    if width > 2048 or height > 2048:
        return "4K"
    if width > 1024 or height > 1024:
        return "2K"
    return "1K"


def _mime_from_bytes(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class GeminiAPI:
    """Stateless-ish REST client for image generation and editing."""

    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = (api_key or "").strip()
        model = (model_name or DEFAULT_MODEL).strip()
        self.model = model[len("models/"):] if model.startswith("models/") else model
        self.prompts = PromptManager.load_prompts()

    # -- model capability heuristics --------------------------------------

    @property
    def _is_gen3(self) -> bool:
        return "gemini-3" in self.model

    @property
    def _supports_image_size(self) -> bool:
        return self._is_gen3 or "pro" in self.model.lower()

    @property
    def _modalities(self) -> List[str]:
        # Gemini 3.x image models accept IMAGE-only responses; the 2.5
        # image family rejects IMAGE-only and requires TEXT+IMAGE.
        return ["IMAGE"] if self._is_gen3 else ["TEXT", "IMAGE"]

    # -- prompt construction ----------------------------------------------

    def _log_prompt(self, mode: str, key: str, system_text: str, user_text: str):
        print(f"\n[GEMINI {mode}] system prompt key: {key}")
        if system_text:
            print(f"[GEMINI {mode}] system prompt applied ({len(system_text)} chars)")
        print(f"[GEMINI {mode}] user prompt: {user_text}\n")

    def _build_generate_prompt(self, user_prompt: str, has_reference: bool,
                               is_color_render: bool) -> str:
        parts = []
        key = None
        system_text = ""
        if get_use_system_prompts():
            if is_color_render:
                key = "color_with_reference" if has_reference else "color_only"
            else:
                key = "depth_with_reference" if has_reference else "depth_only"
            system_text = self.prompts.get(key, "")
            if system_text:
                parts.append(system_text)

        user_prompt = user_prompt.strip()
        if user_prompt:
            parts.append(f"USER_PROMPT (EXECUTE THIS): {user_prompt}")
        elif not parts:
            parts.append(self.prompts.get("default_generate_prompt", "Generate image."))

        self._log_prompt("GENERATE", str(key), system_text, user_prompt or "(default)")
        return "\n\n".join(parts)

    def _build_edit_prompt(self, user_prompt: str, has_mask: bool,
                           has_reference: bool) -> str:
        use_system = get_use_system_prompts()

        if use_system and user_prompt.strip() == "[FINALIZE_COMPOSITE]":
            text = self.prompts.get("finalize_composite", "")
            self._log_prompt("EDIT", "finalize_composite", text, user_prompt)
            return text or "Unify colors, contrast and lighting into one seamless image."

        parts = []
        key = None
        system_text = ""
        if use_system:
            if has_mask:
                key = "inpainting_with_reference" if has_reference else "inpainting_only"
            elif has_reference:
                key = "edit_integration"
            else:
                key = "edit_refinement"
            system_text = self.prompts.get(key, "")
            if system_text:
                parts.append(system_text)

        user_prompt = user_prompt.strip()
        if user_prompt:
            parts.append(f"USER'S EDIT INSTRUCTIONS:\n{user_prompt}")
        elif not parts:
            parts.append(self.prompts.get("default_edit_prompt", "Edit this image."))

        self._log_prompt("EDIT", str(key), system_text, user_prompt or "(default)")
        return "\n\n".join(parts)

    # -- transport ----------------------------------------------------------

    @staticmethod
    def _image_part(path: str) -> Optional[dict]:
        try:
            with open(path, "rb") as f:
                data = f.read()
            return {"inline_data": {
                "mime_type": _mime_from_bytes(data),
                "data": base64.b64encode(data).decode("ascii"),
            }}
        except OSError as e:
            print(f"[GEMINI] Failed to read image '{path}': {e}")
            return None

    def _post(self, payload: dict) -> Tuple[int, dict, str]:
        """POST payload to generateContent. Returns (status, json, raw_text)."""
        if not self.api_key:
            raise GeminiAPIError(
                "No API key configured. Set it in Edit > Preferences > Add-ons > "
                "TextureProjector, or via the GEMINI_API_KEY environment variable.")

        url = f"{BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Client": "blender-texture-projector",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, json.loads(raw), raw
        except HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
                body = json.loads(raw)
            except (ValueError, OSError):
                body = {}
            return e.code, body, raw
        except URLError as e:
            raise GeminiAPIError(f"Network error: {e.reason}")
        except TimeoutError:
            raise GeminiAPIError("Request timed out. Try again or use a smaller resolution.")

    def _describe_http_error(self, status: int, body: dict, raw: str) -> str:
        msg = ""
        if isinstance(body, dict):
            msg = (body.get("error") or {}).get("message", "")
        msg = msg or raw[:300]
        if status in (401, 403):
            return f"API key rejected or quota exhausted ({status}): {msg}"
        if status == 404:
            return (f"Model '{self.model}' not found ({status}). "
                    f"Pick another model in the panel. {msg}")
        if status == 429:
            return f"Rate limited (429). Wait a bit and retry. {msg}"
        if status >= 500:
            return f"Google server error ({status}). Retry later. {msg}"
        return f"API error {status}: {msg}"

    def _request_image(self, parts: List[dict], temperature: float,
                       width: int, height: int) -> Tuple[bytes, str]:
        """Send request with a retry cascade over generationConfig variants."""
        aspect = closest_aspect_ratio(width, height)
        tier = resolution_tier(width, height)

        def make_cfg(image_cfg: Optional[dict], modalities: List[str]) -> dict:
            cfg = {
                "temperature": temperature,
                "maxOutputTokens": 32768,
                "candidateCount": 1,
                "responseModalities": modalities,
            }
            if image_cfg:
                cfg["imageConfig"] = image_cfg
            return cfg

        primary = self._modalities
        flipped = ["TEXT", "IMAGE"] if primary == ["IMAGE"] else ["IMAGE"]

        attempts = []
        if self._supports_image_size:
            attempts.append(make_cfg({"aspectRatio": aspect, "imageSize": tier}, primary))
        attempts.append(make_cfg({"aspectRatio": aspect}, primary))
        attempts.append(make_cfg(None, primary))
        attempts.append(make_cfg(None, flipped))

        last_error = None
        for i, gen_cfg in enumerate(attempts):
            payload = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
            print(f"[GEMINI] Request attempt {i + 1}/{len(attempts)} "
                  f"(model={self.model}, aspect={aspect}, tier={tier})")
            status, body, raw = self._post(payload)

            if status == 200:
                return self._extract_image(body)

            last_error = self._describe_http_error(status, body, raw)
            # Only config-shaped 400s are worth retrying with a simpler config.
            if status == 400:
                print(f"[GEMINI] 400 on attempt {i + 1}, degrading config: {last_error}")
                continue
            raise GeminiAPIError(last_error)

        raise GeminiAPIError(last_error or "All request attempts failed.")

    def _extract_image(self, result: dict) -> Tuple[bytes, str]:
        candidates = result.get("candidates") or []
        if not candidates:
            feedback = result.get("promptFeedback") or {}
            reason = feedback.get("blockReason", "no candidates returned")
            raise GeminiAPIError(f"No image generated ({reason}).")

        candidate = candidates[0]
        content = candidate.get("content")
        if not content:
            reason = candidate.get("finishReason", "UNKNOWN")
            raise GeminiAPIError(f"Response blocked or empty (finish reason: {reason}).")

        texts = []
        for part in content.get("parts", []):
            inline = part.get("inline_data") or part.get("inlineData")
            if inline:
                data = inline.get("data") or inline.get("bytes")
                if data:
                    blob = base64.b64decode(data)
                    return blob, _mime_from_bytes(blob)
            if "text" in part:
                texts.append(part["text"])

        if texts:
            # The model answered with text instead of pixels; surface it.
            snippet = " ".join(texts)[:200]
            raise GeminiAPIError(f"Model returned text instead of an image: {snippet}")
        raise GeminiAPIError("No image data found in the API response.")

    # -- public API ----------------------------------------------------------

    def generate_image(self, depth_image_path: str, user_prompt: str,
                       reference_image_path: str = None,
                       is_color_render: bool = False,
                       width: int = 1024, height: int = 1024) -> Tuple[bytes, str]:
        """Generate a texture from a captured color/depth image."""
        prompt = self._build_generate_prompt(
            user_prompt, has_reference=bool(reference_image_path),
            is_color_render=is_color_render)

        parts = [{"text": prompt}]
        source = self._image_part(depth_image_path)
        if not source:
            raise GeminiAPIError(f"Cannot read capture image: {depth_image_path}")
        parts.append(source)
        if reference_image_path:
            ref = self._image_part(reference_image_path)
            if ref:
                parts.append(ref)

        return self._request_image(parts, temperature=0.8, width=width, height=height)

    def edit_image(self, image_path: str, edit_prompt: str, mask_path: str = None,
                   reference_image_path: str = None,
                   width: int = 0, height: int = 0) -> Tuple[bytes, str]:
        """Edit an existing image, optionally guided by a mask and reference."""
        prompt = self._build_edit_prompt(
            edit_prompt, has_mask=bool(mask_path),
            has_reference=bool(reference_image_path))

        # Order matters: prompt -> reference (style priority) -> original -> mask.
        parts = [{"text": prompt}]
        if reference_image_path:
            ref = self._image_part(reference_image_path)
            if ref:
                parts.append(ref)
        original = self._image_part(image_path)
        if not original:
            raise GeminiAPIError(f"Cannot read image: {image_path}")
        parts.append(original)
        if mask_path:
            mask = self._image_part(mask_path)
            if mask:
                parts.append(mask)

        if width <= 0 or height <= 0:
            width, height = _png_dimensions(image_path) or (1024, 1024)

        return self._request_image(parts, temperature=0.7, width=width, height=height)


def _png_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """Read PNG dimensions from the IHDR header without any image library."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
    except (OSError, struct.error):
        pass
    return None


# ---------------------------------------------------------------------------
# Key / preference access
# ---------------------------------------------------------------------------

def get_prefs():
    """Return this addon's preferences, or None outside Blender."""
    try:
        import bpy
        addon = bpy.context.preferences.addons.get(__package__)
        return addon.preferences if addon else None
    except Exception:
        return None


def get_api_key() -> Optional[str]:
    """Resolve the API key: environment > addon preferences > legacy scene prop."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    prefs = get_prefs()
    if prefs is not None and getattr(prefs, "api_key", "").strip():
        return prefs.api_key.strip()

    # Legacy: old versions stored the key on the scene. Still honored so
    # existing .blend files keep working, but no longer exposed in the UI.
    try:
        import bpy
        props = getattr(bpy.context.scene, "gemini_render", None)
        if props is not None and props.api_key.strip():
            return props.api_key.strip()
    except Exception:
        pass
    return None


def get_use_system_prompts() -> bool:
    prefs = get_prefs()
    if prefs is not None and hasattr(prefs, "use_system_prompts"):
        return bool(prefs.use_system_prompts)
    return True


def validate_api_key_online(api_key: str, timeout: int = 10) -> Tuple[bool, str]:
    """Hit the models list endpoint to verify a key actually works."""
    if not api_key:
        return False, "No API key provided"
    url = f"{BASE_URL}/models?pageSize=1&key={api_key}"
    try:
        req = urllib_request.Request(url, method="GET")
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True, "API key is valid"
            return False, f"Unexpected status {resp.status}"
    except HTTPError as e:
        if e.code in (400, 401, 403):
            return False, f"API key rejected ({e.code})"
        return False, f"HTTP error {e.code}"
    except URLError as e:
        return False, f"Network error: {e.reason}"
    except Exception as e:
        return False, f"Validation failed: {e}"
