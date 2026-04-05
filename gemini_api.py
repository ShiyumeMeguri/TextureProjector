"""
Gemini API integration for image generation using official Python SDK
Unified logic for Texture Projection.
"""

import os
import json
from typing import Optional, Tuple, Dict, Any
from io import BytesIO

# I try importing PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
    print(" PIL (Pillow) available")
except ImportError:
    print(" PIL not installed - some features will use fallback")
    PIL_AVAILABLE = False

# I try importing official Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    print(" Official google-genai SDK available")
except ImportError:
    print(" google-genai not installed, using fallback REST API")
    GENAI_AVAILABLE = False

# Import requests for REST API fallback
try:
    import requests
    import base64
except ImportError:
    print(" Warning: requests library not found. REST API fallback will fail.")

SYSTEM_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "system_prompts.json")

class PromptManager:
    """Handles loading and accessing system prompts from JSON configuration."""
    
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
        "default_generate_prompt": "Generate image."
    }

    @classmethod
    def load_prompts(cls) -> Dict[str, str]:
        """Load prompts from JSON file or return defaults if file missing/invalid."""
        if not os.path.exists(SYSTEM_PROMPTS_FILE):
            print(f" System prompts file not found at {SYSTEM_PROMPTS_FILE}, using defaults.")
            # Optional: Create the file if it doesn't exist so user can edit it later?
            # For now, just return defaults.
            return cls.DEFAULT_PROMPTS.copy()
            
        try:
            with open(SYSTEM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Merge with defaults to ensure all keys exist
            prompts = cls.DEFAULT_PROMPTS.copy()
            prompts.update(data)
            return prompts
        except Exception as e:
            print(f" Error loading system prompts: {e}. Using defaults.")
            return cls.DEFAULT_PROMPTS.copy()

class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors"""
    pass

class GeminiAPI:
    """Client for Google Gemini API with official SDK"""
    
    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = api_key
        self.model = model_name if model_name else "gemini-2.5-flash-image"
        
        # I ensure model has 'models/' prefix for REST if missing
        if self.model.startswith("models/"):
             self.rest_model = self.model
             self.base_model = self.model.replace("models/", "")
        else:
             self.rest_model = f"models/{self.model}"
             self.base_model = self.model

        if GENAI_AVAILABLE and PIL_AVAILABLE:
            print(f" Using official Google GenAI SDK (Model: {self.model})")
            try:
                # I configure the official client
                # New SDK (google-genai) passes api_key to Client constructor
                self.client = genai.Client(api_key=api_key)
                self.use_sdk = True
            except Exception as e:
                print(f" SDK setup failed: {e}, falling back to REST")
                self.use_sdk = False
                self._setup_rest_fallback()
        else:
            if not GENAI_AVAILABLE:
                print(" google-genai SDK not available, using REST API fallback")
            elif not PIL_AVAILABLE:
                print(" PIL not available, using REST API fallback (SDK requires PIL)")
            self.use_sdk = False
            self._setup_rest_fallback()
        
        # Load system prompts
        self.prompts = PromptManager.load_prompts()
    
    def _setup_rest_fallback(self):
        """Setup REST API fallback"""
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        # model is already set in __init__
        
    def _log_prompt_debug(self, prompt_type: str, key: str, system_text: str, user_text: str):
        """Strictly log prompt construction for verification"""
        # Translate types
        cn_type = "生成模式 (GENERATION)" if prompt_type == "GENERATION" else "编辑模式 (EDIT)"
        
        print(f"\n{'='*20} [GEMINI {cn_type} 提示词调试信息] {'='*20}")
        print(f"🔑 当前生效模式 (Current Mode): 【 {key} 】")
        
        if system_text:
            print(f"📅 系统提示词完整内容 (System Prompt Full Content):")
            print(f"{'-'*60}")
            print(f"{system_text}") # PRINT FULL CONTENT
            print(f"{'-'*60}")
        else:
             print(f"❌ 未应用系统提示词 (No System Prompt)")

        print(f"👤 用户最终指令 (User Instruction):")
        print(f"{'-'*60}")
        print(f"{user_text}")
        print(f"{'-'*60}")
        print(f"{'='*60}\n")

    def _build_prompt(self, user_prompt: str, has_reference: bool = False, is_color_render: bool = False) -> str:
        """Build prompt based on mode (Depth vs Color)"""
        prompt_parts = []
        
        # Check preference
        use_system = get_use_system_prompts()
        
        prompt_key = None
        system_text = ""
        
        # SYSTEM PROMPT SELECTION
        if use_system:
            if not is_color_render:
                # DEPTH MODE
                if has_reference:
                    prompt_key = "depth_with_reference"
                else:
                    prompt_key = "depth_only"
            else:
                # COLOR/SCREENSHOT MODE
                if has_reference:
                    prompt_key = "color_with_reference"
                else:
                    prompt_key = "color_only"
            
            if prompt_key:
                system_text = self.prompts.get(prompt_key, "")
                if system_text:
                    prompt_parts.append(system_text)
        
        # User Prompt Assembly
        final_user_text = ""
        if user_prompt.strip():
            # Match Reference Style: Explicit Label
            final_user_text = f"USER_PROMPT (EXECUTE THIS): {user_prompt.strip()}"
            prompt_parts.append(final_user_text)
        else:
            if not prompt_parts: # If no system prompt and no user prompt
                default_prompt = self.prompts.get("default_generate_prompt", "Generate image.")
                prompt_parts.append(default_prompt)
                final_user_text = f"(默认) {default_prompt}"
        
        # Debug Log
        self._log_prompt_debug("GENERATION", str(prompt_key), system_text, user_prompt.strip() or "(用户未输入 - 使用默认值)")
            
        return "\n\n".join(prompt_parts)
    
    def _build_edit_prompt(self, user_prompt: str, has_mask: bool = False, has_reference: bool = False) -> str:
        """Build modular edit prompt based on input data types (Mask vs Reference vs None)"""
        prompt_parts = []
        
        # Check preference
        use_system = get_use_system_prompts()
        
        prompt_key = None
        system_text = ""
        
        # Logic from user's framework
        # 1. Special case: Finalize Composite
        if use_system and user_prompt.strip() == "[FINALIZE_COMPOSITE]":
             prompt_key = "finalize_composite"
             system_text = self.prompts.get(prompt_key, "")
             self._log_prompt_debug("EDIT", prompt_key, system_text, user_prompt)
             return system_text
             
        # Select prompt key based on inputs
        if use_system:
            if has_mask:
                # INPAINTING MODES
                if has_reference:
                    prompt_key = "inpainting_with_reference"
                else:
                    prompt_key = "inpainting_only"
            elif has_reference:
                # INTEGRATION MODE (Reference but no mask)
                prompt_key = "edit_integration"
            else:
                # REFINEMENT MODE (No mask, no reference - just existing image + prompt)
                prompt_key = "edit_refinement"
        
        # Add system prompt if found
        if prompt_key:
            system_text = self.prompts.get(prompt_key, "")
            if system_text:
                prompt_parts.append(system_text)
            
        final_user_text = ""
        if user_prompt.strip():
             # Match Reference Style: Explicit Label
             final_user_text = f"USER'S EDIT INSTRUCTIONS:\n{user_prompt.strip()}"
             prompt_parts.append(final_user_text)
        else:
             # Default Fallback
             default_prompt = self.prompts.get("default_edit_prompt", "Edit this image.")
             prompt_parts.append(default_prompt)
             final_user_text = f"(Default) {default_prompt}"
        
        # Debug Log
        self._log_prompt_debug("EDIT", str(prompt_key), system_text, user_prompt.strip() or "(Empty User Prompt - Using Default)")
        
        return "\n\n".join(prompt_parts)
    
    # =========================================================================
    # CORE LOGIC: Parameter Calculation (Unified)
    # =========================================================================
    
    def _get_image_config_params(self, width: int, height: int) -> Tuple[str, Optional[str]]:
        """
        Unified logic to determine Aspect Ratio and Image Size.
        Returns: (aspect_ratio_string, image_size_string_or_None)
        """
        # 1. Calculate Aspect Ratio (Always needed)
        if width <= 0 or height <= 0:
            target_ratio = 1.0
        else:
            target_ratio = width / height
            
        ratios = {
            "1:1": 1.0,
            "2:3": 2/3, "3:2": 3/2,
            "3:4": 3/4, "4:3": 4/3,
            "4:5": 4/5, "5:4": 5/4,
            "9:16": 9/16, "16:9": 16/9,
            "21:9": 21/9
        }
        # Find closest aspect ratio
        aspect_ratio_str = min(ratios.keys(), key=lambda k: abs(ratios[k] - target_ratio))
        
        # 2. Determine Image Size (Only for Pro/V3 models)
        image_size_str = None
        is_pro = "pro" in self.model.lower() or "gemini-3" in self.model.lower()
        
        if is_pro:
            # Logic based on documentation (1K, 2K, 4K)
            # Thresholds: >2048 triggers 4K, >1024 triggers 2K
            if width > 2048 or height > 2048:
                image_size_str = "4K"
            elif width > 1024 or height > 1024:
                image_size_str = "2K"
            else:
                image_size_str = "1K"
                
        return aspect_ratio_str, image_size_str

    # =========================================================================
    # GENERATION METHODS
    # =========================================================================

    def generate_image(self, depth_image_path: str, user_prompt: str, reference_image_path: str = None, is_color_render: bool = False, width: int = 1024, height: int = 1024) -> Tuple[bytes, str]:
        """Generate image from depth/color map and prompt"""
        if self.use_sdk:
            return self._generate_with_sdk(depth_image_path, user_prompt, reference_image_path, is_color_render, width, height)
        else:
            return self._generate_with_rest(depth_image_path, user_prompt, reference_image_path, is_color_render, width, height)
    
    def _generate_with_sdk(self, depth_image_path: str, user_prompt: str, reference_image_path: str = None, is_color_render: bool = False, width: int = 1024, height: int = 1024) -> Tuple[bytes, str]:
        try:
            if not PIL_AVAILABLE:
                self.use_sdk = False
                self._setup_rest_fallback()
                return self._generate_with_rest(depth_image_path, user_prompt, reference_image_path, is_color_render, width, height)
            
            # Prepare Inputs
            full_prompt = self._build_prompt(user_prompt, has_reference=bool(reference_image_path), is_color_render=is_color_render)
            contents = [full_prompt, Image.open(depth_image_path)]
            if reference_image_path:
                try:
                    contents.append(Image.open(reference_image_path))
                except Exception as e:
                    print(f" Failed to load reference: {e}")
            
            # Unified Config Logic
            aspect_ratio, image_size = self._get_image_config_params(width, height)
            # Unified Config Logic
            aspect_ratio, image_size = self._get_image_config_params(width, height)
            print(f"🚀 [SDK] Prompting with Aspect Ratio: {aspect_ratio}, Size: {image_size}")
            print(f"📝 Full System Prompt: {full_prompt}")
            
            try:
                # Build ImageConfig object
                img_conf_args = {"aspect_ratio": aspect_ratio}
                if image_size:
                    img_conf_args["image_size"] = image_size
                
                config = types.GenerateContentConfig(
                    temperature=0.8,
                    candidate_count=1,
                    response_modalities=['TEXT', 'IMAGE'],
                    image_config=types.ImageConfig(**img_conf_args)
                )
            except Exception as e:
                print(f" Config setup warning: {e}")
                config = types.GenerateContentConfig(temperature=0.8, response_modalities=['TEXT', 'IMAGE'])

            # Call API
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            return self._process_sdk_response(response)
                
        except Exception as e:
            if isinstance(e, GeminiAPIError): raise
            print(f" SDK error: {e}, falling back to REST")
            self.use_sdk = False
            self._setup_rest_fallback()
            return self._generate_with_rest(depth_image_path, user_prompt, reference_image_path, is_color_render, width, height)
    
    def _generate_with_rest(self, depth_image_path: str, user_prompt: str, reference_image_path: str = None, is_color_render: bool = False, width: int = 1024, height: int = 1024) -> Tuple[bytes, str]:
        try:
            # Prepare Inputs
            full_prompt = self._build_prompt(user_prompt, has_reference=bool(reference_image_path), is_color_render=is_color_render)
            parts = [{"text": full_prompt}, self._load_image_part_rest(depth_image_path)]
            if reference_image_path:
                ref_part = self._load_image_part_rest(reference_image_path)
                if ref_part: parts.append(ref_part)
            
            # Unified Config Logic
            aspect_ratio, image_size = self._get_image_config_params(width, height)
            # Unified Config Logic
            aspect_ratio, image_size = self._get_image_config_params(width, height)
            print(f"🚀 [REST] Prompting with Aspect Ratio: {aspect_ratio}, Size: {image_size}")
            print(f"📝 Full System Prompt: {full_prompt}")
            
            # Build Payload Helper
            def _build_payload(use_size: bool = True):
                gen_cfg = {
                    "temperature": 0.8,
                    "maxOutputTokens": 32768,
                    "candidateCount": 1,
                    "responseModalities": ["TEXT", "IMAGE"],
                }
                # Always send aspectRatio, conditionally send imageSize
                img_cfg = {"aspectRatio": aspect_ratio}
                if use_size and image_size:
                    img_cfg["imageSize"] = image_size
                
                gen_cfg["imageConfig"] = img_cfg
                return {"contents": [{"parts": parts}], "generationConfig": gen_cfg}

            # Call API
            url = f"{self.base_url}/{self.rest_model}:generateContent?key={self.api_key}"
            headers = {'Content-Type': 'application/json', 'X-Goog-Api-Client': 'python-blender-addon'}
            
            response = requests.post(url, headers=headers, json=_build_payload(True), timeout=300)
            
            # Retry logic: if failed due to config, try without imageSize but KEEP aspectRatio
            if response.status_code == 400 and ("imageSize" in response.text or "imageConfig" in response.text):
                print(" Model rejected config, retrying without imageSize...")
                response = requests.post(url, headers=headers, json=_build_payload(False), timeout=300)
                
            return self._process_rest_response(response)
            
        except Exception as e:
            if isinstance(e, GeminiAPIError): raise
            raise GeminiAPIError(f"Unexpected error: {str(e)}")

    # =========================================================================
    # EDIT METHODS
    # =========================================================================

    def edit_image(self, image_path: str, edit_prompt: str, mask_path: str = None, reference_image_path: str = None, width: int = 0, height: int = 0) -> Tuple[bytes, str]:
        """Edit existing image"""
        try:
            full_prompt = self._build_edit_prompt(edit_prompt, bool(mask_path), bool(reference_image_path))
            print(f"Starting edit. Model: {self.model}, Prompt: {full_prompt}")
            
            if self.use_sdk:
                return self._edit_with_sdk(image_path, full_prompt, mask_path, reference_image_path, width, height)
            else:
                return self._edit_with_rest(image_path, full_prompt, mask_path, reference_image_path, width, height)
        except Exception as e:
            if isinstance(e, GeminiAPIError): raise
            raise GeminiAPIError(f"Edit failed: {str(e)}")
    
    def _edit_with_sdk(self, image_path: str, prompt: str, mask_path: str = None, reference_path: str = None, width: int = 0, height: int = 0) -> Tuple[bytes, str]:
        try:
            if not PIL_AVAILABLE:
                self.use_sdk = False
                self._setup_rest_fallback()
                return self._edit_with_rest(image_path, prompt, mask_path, reference_path, width, height)
            
            # Prepare Inputs
            orig_img = Image.open(image_path)
            contents = [prompt]
            if reference_path: contents.append(Image.open(reference_path))
            contents.append(orig_img)
            if mask_path:
                mask_img = Image.open(mask_path)
                if mask_img.mode != 'L': mask_img = mask_img.convert('L')
                contents.append(mask_img)
            
            # Unified Config Logic
            # If width/height not provided (0), use image size
            target_w = width if width > 0 else orig_img.size[0]
            target_h = height if height > 0 else orig_img.size[1]
            
            aspect_ratio, image_size = self._get_image_config_params(target_w, target_h)
            
            try:
                img_conf_args = {"aspect_ratio": aspect_ratio}
                if image_size:
                    img_conf_args["image_size"] = image_size
                    
                config = types.GenerateContentConfig(
                    temperature=0.7,
                    candidate_count=1,
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(**img_conf_args)
                )
            except Exception as e:
                config = types.GenerateContentConfig(temperature=0.7, response_modalities=['IMAGE'])
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            return self._process_sdk_response(response)
            
        except Exception as e:
            if isinstance(e, GeminiAPIError): raise
            print(f"SDK edit error: {e}, falling back to REST")
            self.use_sdk = False
            self._setup_rest_fallback()
            return self._edit_with_rest(image_path, prompt, mask_path, reference_path, width, height)

    def _edit_with_rest(self, image_path: str, prompt: str, mask_path: str = None, reference_path: str = None, width: int = 0, height: int = 0) -> Tuple[bytes, str]:
        try:
            parts = [{"text": prompt}]
            if reference_path:
                p = self._load_image_part_rest(reference_path)
                if p: parts.append(p)
            parts.append(self._load_image_part_rest(image_path))
            if mask_path:
                p = self._load_image_part_rest(mask_path)
                if p: parts.append(p)
            
            # Resolution resolution
            # If width/height not passed, we need to guess or read file. 
            # For REST fallback without PIL, we might not know size if 0 passed.
            # Assuming callers pass valid width/height or we rely on default 1:1 fallback in helper.
            aspect_ratio, image_size = self._get_image_config_params(width, height)
            
            def _build_payload(use_size: bool = True):
                gen_cfg = {
                    "temperature": 0.7,
                    "maxOutputTokens": 32768,
                    "candidateCount": 1,
                    "responseModalities": ["TEXT", "IMAGE"],
                }
                img_cfg = {"aspectRatio": aspect_ratio}
                if use_size and image_size:
                    img_cfg["imageSize"] = image_size
                gen_cfg["imageConfig"] = img_cfg
                return {"contents": [{"parts": parts}], "generationConfig": gen_cfg}

            url = f"{self.base_url}/{self.rest_model}:generateContent?key={self.api_key}"
            headers = {'Content-Type': 'application/json', 'X-Goog-Api-Client': 'python-blender-addon'}
            
            response = requests.post(url, headers=headers, json=_build_payload(True), timeout=300)
            
            if response.status_code == 400 and ("imageSize" in response.text or "imageConfig" in response.text):
                 print(" Model rejected config, retrying without imageSize...")
                 response = requests.post(url, headers=headers, json=_build_payload(False), timeout=300)
            
            return self._process_rest_response(response)

        except requests.RequestException as e:
            raise GeminiAPIError(f"Network error: {str(e)}")
        except Exception as e:
            if isinstance(e, GeminiAPIError): raise
            raise GeminiAPIError(f"Edit failed: {str(e)}")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _load_image_part_rest(self, path: str) -> Optional[Dict]:
        try:
            with open(path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            return {"inline_data": {"mime_type": "image/png", "data": b64}}
        except:
            return None



    def _process_sdk_response(self, response) -> Tuple[bytes, str]:
        if not response.candidates:
            raise GeminiAPIError("No candidates returned from API.")
        
        candidate = response.candidates[0]
        
        # Safety Check: Content might be None if blocked
        if not candidate.content:
            reason = "Unknown"
            try: reason = str(candidate.finish_reason)
            except: pass
            raise GeminiAPIError(f"Response blocked. Finish Reason: {reason}")

        if not candidate.content.parts:
            raise GeminiAPIError("No content parts generated.")
        
        for part in candidate.content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                if image.mode not in ('RGB', 'RGBA'): image = image.convert('RGB')
                buf = BytesIO()
                image.save(buf, format='PNG')
                return buf.getvalue(), "image/png"
        
        text_parts = [p.text for p in candidate.content.parts if p.text]
        if text_parts:
            return self._create_placeholder_image(f"Response: {' '.join(text_parts)}")
        raise GeminiAPIError("No image data returned")

    def _process_rest_response(self, response) -> Tuple[bytes, str]:
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_msg = error_json['error'].get('message', error_msg)
            except: pass
            raise GeminiAPIError(f"API Error {response.status_code}: {error_msg}")
        
        result = response.json()
        if 'candidates' not in result or not result['candidates']:
            raise GeminiAPIError("No candidates.")
        
        candidate = result['candidates'][0]
        
        # Safety/Content Check
        if 'content' not in candidate:
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            raise GeminiAPIError(f"Response blocked or empty. Finish Reason: {finish_reason}")
        
        parts = candidate['content'].get('parts', [])
        for part in parts:
            inline = part.get('inline_data') or part.get('inlineData')
            if inline:
                data = inline.get('data') or inline.get('bytes')
                if data:
                    return base64.b64decode(data), inline.get('mime_type', 'image/png')
        
        text_parts = [p.get('text', '') for p in parts if 'text' in p]
        if text_parts:
            return self._create_placeholder_image(f"Response: {' '.join(text_parts)}")
        
        raise GeminiAPIError("No image found in response")

    def _create_placeholder_image(self, text_response: str) -> Tuple[bytes, str]:
        # Simple blue 100x100 png
        return self._create_simple_png(100, 100, (0, 100, 200)), "image/png"

    def _create_simple_png(self, width: int, height: int, color: tuple) -> bytes:
        import zlib, struct
        png_sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        ihdr = struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0)
        ihdr_c = zlib.crc32(b'IHDR' + ihdr) & 0xffffffff
        ihdr_chunk = struct.pack('>I', len(ihdr)) + b'IHDR' + ihdr + struct.pack('>I', ihdr_c)
        raw = b''
        for _ in range(height):
            raw += b'\x00' + struct.pack('BBB', *color) * width
        idat = zlib.compress(raw)
        idat_c = zlib.crc32(b'IDAT' + idat) & 0xffffffff
        idat_chunk = struct.pack('>I', len(idat)) + b'IDAT' + idat + struct.pack('>I', idat_c)
        iend_c = zlib.crc32(b'IEND') & 0xffffffff
        iend_chunk = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_c)
        return png_sig + ihdr_chunk + idat_chunk + iend_chunk

def get_api_key() -> Optional[str]:
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key: return api_key
    import bpy
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        if hasattr(prefs, 'api_key') and prefs.api_key.strip(): return prefs.api_key.strip()
    except: pass
    try:
        if hasattr(bpy.context.scene, 'gemini_render') and bpy.context.scene.gemini_render.api_key.strip():
            return bpy.context.scene.gemini_render.api_key.strip()
    except: pass
    return None

def get_use_system_prompts() -> bool:
    """Check if system prompts are enabled in preferences (default: True)"""
    import bpy
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        if hasattr(prefs, 'use_system_prompts'):
            return prefs.use_system_prompts
    except:
        pass
    return True
