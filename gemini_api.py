"""
Gemini API integration for image generation using official Python SDK
Minimalist version for Texture Projection - No heavy system prompts.
"""

import os
from typing import Optional, Tuple
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

    import requests
    import json
    import base64

class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors"""
    pass

class GeminiAPI:
    """Client for Google Gemini API with official SDK"""
    
    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = api_key
        # I default model if not provided
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
                genai.configure(api_key=api_key)
                self.client = genai.Client()
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
    
    def _setup_rest_fallback(self):
        """Setup REST API fallback"""
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        # model is already set in __init__
        
    def _build_prompt(self, user_prompt: str, has_reference: bool = False, is_color_render: bool = False) -> str:
        """
        Build minimal prompt.
        Removed all 'rendering' system instructions.
        Now purely passes user intent for texture projection.
        """
        prompt_parts = []

        # 1. Basic role/intent (Optional, keeps it strictly technical)
        # prompt_parts.append("Generate a texture/image based on the input structure.")

        # 2. Reference handling (Minimal instruction)
        if has_reference:
            prompt_parts.append("Use the provided reference image for style, color, and texture details.")

        # 3. The User Prompt (The only thing that matters now)
        if user_prompt.strip():
            prompt_parts.append(user_prompt.strip())
        else:
            prompt_parts.append("Generate image.")

        # 4. Technical constraint (Optional, but usually good for projection)
        # prompt_parts.append("Maintain the exact geometry and layout of the input image.")

        return "\n\n".join(prompt_parts)
    
    def _build_edit_prompt(self, user_prompt: str, has_mask: bool = False, has_reference: bool = False) -> str:
        """
        Build minimal edit prompt.
        Removed all 'composite/repair' system instructions.
        """
        # Pure pass-through of user instructions.
        # The model infers task from the presence of mask/reference images in the payload.
        
        final_prompt = user_prompt.strip()
        
        if not final_prompt:
            final_prompt = "Edit this image."
            
        return final_prompt
    
    def generate_image(self, depth_image_path: str, user_prompt: str, reference_image_path: str = None, is_color_render: bool = False, width: int = 1024, height: int = 1024) -> Tuple[bytes, str]:
        """
        Generate image from depth map and prompt using official SDK
        """
        if self.use_sdk:
            return self._generate_with_sdk(depth_image_path, user_prompt, reference_image_path, is_color_render, width, height)
        else:
            return self._generate_with_rest(depth_image_path, user_prompt, reference_image_path, is_color_render, width, height)
    
    def _generate_with_sdk(self, depth_image_path: str, user_prompt: str, reference_image_path: str = None, is_color_render: bool = False, width: int = 1024, height: int = 1024) -> Tuple[bytes, str]:
        """Generate image using official Google GenAI SDK"""
        try:
            if not PIL_AVAILABLE:
                print(" PIL not available for SDK, switching to REST")
                self.use_sdk = False
                self._setup_rest_fallback()
                return self._generate_with_rest(depth_image_path, user_prompt, reference_image_path, is_color_render, width, height)
            
            # I build complete prompt
            full_prompt = self._build_prompt(user_prompt, has_reference=bool(reference_image_path), is_color_render=is_color_render)
            
            # --- PRINT FINAL PAYLOAD ---
            print("\n" + "="*60)
            print(f"🚀 [GEMINI API] FINAL GENERATE PAYLOAD (SDK) - CLEAN:")
            print(f"📝 PROMPT: {full_prompt}")
            print("-" * 30)
            print(f"📂 INPUT IMAGE: {depth_image_path}")
            if reference_image_path:
                print(f"📂 REFERENCE: {reference_image_path}")
            print("="*60 + "\n")
            # ---------------------------
            
            # I load depth image using PIL
            depth_image = Image.open(depth_image_path)
            
            # I prepare contents for the API call
            contents = [full_prompt]
            contents.append(depth_image)
            
            if reference_image_path:
                try:
                    reference_image = Image.open(reference_image_path)
                    contents.append(reference_image)
                except Exception as e:
                    print(f" Failed to load reference image: {e}")
            
            # I map resolution
            resolution_str = "1K"
            if width >= 4096 or height >= 4096:
                resolution_str = "4K"
            elif width >= 2048 or height >= 2048:
                resolution_str = "2K"
            
            try:
                # Basic config
                config = types.GenerateContentConfig(
                    temperature=0.8,
                    candidate_count=1,
                    response_modalities=['TEXT', 'IMAGE']
                )
                
                # Try to add image config if available
                if hasattr(types, 'ImageConfig'):
                    img_conf = types.ImageConfig(
                        image_size=resolution_str,
                        # [FIX] Removed aspect_ratio="1:1" to allow non-square resolutions from threading_utils
                    )
                    config.image_config = img_conf
                else:
                    # Generic dict fallback
                    pass # SDK might handle raw dicts differently, keeping it simple for now

            except Exception as e:
                print(f" Config setup warning: {e}")
                config = types.GenerateContentConfig(
                    temperature=0.8,
                    candidate_count=1,
                    response_modalities=['TEXT', 'IMAGE']
                )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            # I process response parts
            if not response.candidates or not response.candidates[0].content.parts:
                raise GeminiAPIError("No image generated.")
            
            parts = response.candidates[0].content.parts

            for part in parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    if image.mode not in ('RGB', 'RGBA'):
                        image = image.convert('RGB')
                    
                    img_byte_arr = BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    return img_byte_arr.getvalue(), "image/png"
            
            text_parts = [part.text for part in parts if part.text is not None]
            if text_parts:
                return self._create_placeholder_image(f"Model response: {' '.join(text_parts)}")
            else:
                raise GeminiAPIError("No image data returned")
                
        except Exception as e:
            if isinstance(e, GeminiAPIError):
                raise
            print(f" SDK error: {str(e)}, falling back to REST")
            self.use_sdk = False
            self._setup_rest_fallback()
            return self._generate_with_rest(depth_image_path, user_prompt, reference_image_path, is_color_render)
    
    def _generate_with_rest(self, depth_image_path: str, user_prompt: str, reference_image_path: str = None, is_color_render: bool = False, width: int = 1024, height: int = 1024) -> Tuple[bytes, str]:
        """Generate image using REST API fallback"""
        try:
            # I encode images
            with open(depth_image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            reference_base64 = None
            if reference_image_path:
                try:
                    with open(reference_image_path, 'rb') as f:
                        reference_base64 = base64.b64encode(f.read()).decode('utf-8')
                except Exception:
                    pass
            
            # I build complete prompt
            full_prompt = self._build_prompt(user_prompt, has_reference=bool(reference_image_path), is_color_render=is_color_render)
            
            # --- PRINT FINAL PAYLOAD ---
            print("\n" + "="*60)
            print(f"🚀 [GEMINI API] FINAL GENERATE PAYLOAD (REST) - CLEAN:")
            print(f"📝 PROMPT: {full_prompt}")
            print("-" * 30)
            print(f"📂 INPUT IMAGE: {depth_image_path}")
            if reference_image_path:
                print(f"📂 REFERENCE: {reference_image_path}")
            print("="*60 + "\n")
            # ---------------------------
            
            # I prepare REST API request
            model_path = self.rest_model
            if not model_path.startswith("models/"):
                 model_path = f"models/{model_path}"
                 
            url = f"{self.base_url}/{model_path}:generateContent?key={self.api_key}"
            
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Client': 'python-blender-addon',
            }
            
            parts = [{"text": full_prompt}]
            
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": image_base64
                }
            })
            
            if reference_base64:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": reference_base64
                    }
                })
            
            resolution_str = "1K"
            if width >= 4096 or height >= 4096:
                resolution_str = "4K"
            elif width >= 2048 or height >= 2048:
                resolution_str = "2K"
            
            is_pro = "pro" in self.model.lower() or "gemini-3" in self.model.lower()
            
            def _build_payload(res_str: str = None):
                gen_cfg = {
                    "temperature": 0.8,
                    "maxOutputTokens": 32768,
                    "candidateCount": 1,
                    "responseModalities": ["TEXT", "IMAGE"],
                }
                if res_str:
                    gen_cfg["imageConfig"] = {
                        "imageSize": res_str,
                        # [FIX] Removed aspectRatio to allow API to infer from input image
                    }
                return {
                    "contents": [{"parts": parts}],
                    "generationConfig": gen_cfg
                }

            payload = _build_payload(resolution_str if is_pro else None)
            
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            
            # Retry without imageSize if it fails (fallback for some models)
            if response.status_code == 400 and ("imageSize" in response.text or "imageConfig" in response.text):
                print(" Model doesn't support imageSize, retrying without it...")
                payload = _build_payload(None)
                response = requests.post(url, headers=headers, json=payload, timeout=300)

            if response.status_code != 200:
                raise GeminiAPIError(f"API request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if 'candidates' not in result or not result['candidates']:
                raise GeminiAPIError("No candidates in response")
            
            candidate = result['candidates'][0]
            if 'content' not in candidate:
                raise GeminiAPIError("No content in candidate")
            
            parts = candidate['content']['parts'] 
            
            for part in parts:
                inline_data_key = None
                if 'inline_data' in part: inline_data_key = 'inline_data'
                elif 'inlineData' in part: inline_data_key = 'inlineData'
                
                if inline_data_key:
                    inline_data = part[inline_data_key]
                    data_key = 'data' if 'data' in inline_data else 'bytes' if 'bytes' in inline_data else None
                    
                    if data_key and inline_data[data_key]:
                        image_data = base64.b64decode(inline_data[data_key])
                        mime_type = inline_data.get('mime_type', 'image/png')
                        return image_data, mime_type
            
            text_parts = [part.get('text', '') for part in parts if 'text' in part]
            if text_parts:
                return self._create_placeholder_image(f"Model response: {' '.join(text_parts)}")
            
            raise GeminiAPIError("No image data found in API response")
            
        except Exception as e:
            if isinstance(e, GeminiAPIError):
                raise
            raise GeminiAPIError(f"Unexpected error: {str(e)}")
    
    def _create_placeholder_image(self, text_response: str) -> Tuple[bytes, str]:
        """Create a placeholder image with text info"""
        try:
            # I simple 100x100 colored PNG
            width, height = 100, 100
            png_data = self._create_simple_png(width, height, (0, 100, 200))  # Blue
            return png_data, "image/png"
        except Exception as e:
            raise GeminiAPIError(f"Failed to create placeholder: {str(e)}")
    
    def _create_simple_png(self, width: int, height: int, color: tuple) -> bytes:
        """Create a simple colored PNG"""
        import zlib
        import struct
        
        # PNG signature
        png_signature = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        ihdr_data = struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr_chunk = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        
        raw_data = b''
        r, g, b = color
        for y in range(height):
            raw_data += b'\x00'
            for x in range(width):
                raw_data += struct.pack('BBB', r, g, b)
        
        compressed_data = zlib.compress(raw_data)
        idat_crc = zlib.crc32(b'IDAT' + compressed_data) & 0xffffffff  
        idat_chunk = struct.pack('>I', len(compressed_data)) + b'IDAT' + compressed_data + struct.pack('>I', idat_crc)
        
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend_chunk = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        
        return png_signature + ihdr_chunk + idat_chunk + iend_chunk
    
    def edit_image(self, 
                   image_path: str, 
                   edit_prompt: str, 
                   mask_path: str = None,
                   reference_image_path: str = None,
                   width: int = 0,
                   height: int = 0) -> Tuple[bytes, str]:
        """
        Edit existing image with AI based on prompt and optional mask
        """
        try:
            print(f"Starting image edit with model: {self.model}")
            
            # I build edit prompt - CLEAN version
            full_prompt = self._build_edit_prompt(
                edit_prompt, 
                has_mask=bool(mask_path),
                has_reference=bool(reference_image_path)
            )
            
            # --- PRINT FINAL EDIT PAYLOAD ---
            print("\n" + "="*60)
            print(f"🚀 [GEMINI API] FINAL EDIT PAYLOAD - CLEAN:")
            print(f"📝 PROMPT: {full_prompt}")
            print("-" * 30)
            print(f"📂 ORIGINAL IMAGE: {image_path}")
            if mask_path:
                print(f"📂 MASK IMAGE: {mask_path}")
            if reference_image_path:
                print(f"📂 REFERENCE IMAGE: {reference_image_path}")
            print("="*60 + "\n")
            # --------------------------------
            
            if self.use_sdk:
                return self._edit_with_sdk(image_path, full_prompt, mask_path, reference_image_path, width, height)
            else:
                return self._edit_with_rest(image_path, full_prompt, mask_path, reference_image_path, width, height)
        
        except Exception as e:
            if isinstance(e, GeminiAPIError):
                raise
            raise GeminiAPIError(f"Image edit failed: {str(e)}")
    
    def _edit_with_sdk(self, image_path: str, prompt: str, mask_path: str = None, reference_path: str = None, width: int = 0, height: int = 0) -> Tuple[bytes, str]:
        """Edit image using SDK"""
        try:
            if not PIL_AVAILABLE:
                print("PIL not available, switching to REST")
                self.use_sdk = False
                self._setup_rest_fallback()
                return self._edit_with_rest(image_path, prompt, mask_path, reference_path, width, height)
            
            original_image = Image.open(image_path)
            
            # I build contents
            contents = [prompt]
            
            if reference_path:
                reference_image = Image.open(reference_path)
                contents.append(reference_image)
            
            contents.append(original_image)
            
            if mask_path:
                mask_image = Image.open(mask_path)
                if mask_image.mode != 'L':
                    mask_image = mask_image.convert('L')
                contents.append(mask_image)
            
            # Resolution logic
            resolution_str = "1K"
            if width > 0 and height > 0:
                if width >= 4096 or height >= 4096: resolution_str = "4K"
                elif width >= 2048 or height >= 2048: resolution_str = "2K"
            else:
                w, h = original_image.size
                if w >= 4096 or h >= 4096: resolution_str = "4K"
                elif w >= 2048 or h >= 2048: resolution_str = "2K"
                
            try:
                if hasattr(types, 'ImageConfig'):
                    img_conf = types.ImageConfig(image_size=resolution_str) # [FIX] Removed aspect_ratio="1:1"
                    config = types.GenerateContentConfig(
                        temperature=0.7,
                        candidate_count=1,
                        response_modalities=['IMAGE'],
                        image_config=img_conf
                    )
                else:
                    # Dictionary fallback not fully implemented in SDK yet, keep generic
                    config = types.GenerateContentConfig(temperature=0.7, candidate_count=1, response_modalities=['IMAGE'])
            except Exception as e:
                config = types.GenerateContentConfig(temperature=0.7, candidate_count=1, response_modalities=['IMAGE'])
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            if not response.candidates or not response.candidates[0].content.parts:
                raise GeminiAPIError("No content in edit response")
            
            parts = response.candidates[0].content.parts
            for part in parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    if image.mode not in ('RGB', 'RGBA'):
                        image = image.convert('RGB')
                    img_byte_arr = BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    return img_byte_arr.getvalue(), "image/png"
            
            raise GeminiAPIError("No image found in edit response")
            
        except Exception as e:
            if isinstance(e, GeminiAPIError):
                raise
            print(f"SDK edit error: {e}, falling back to REST")
            self.use_sdk = False
            self._setup_rest_fallback()
            return self._edit_with_rest(image_path, prompt, mask_path, reference_path)
    
    def _edit_with_rest(self, image_path: str, prompt: str, mask_path: str = None, reference_path: str = None, width: int = 0, height: int = 0) -> Tuple[bytes, str]:
        """Edit image using REST API"""
        try:
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            parts = [{"text": prompt}]
            
            if reference_path:
                with open(reference_path, 'rb') as f:
                    reference_base64 = base64.b64encode(f.read()).decode('utf-8')
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": reference_base64
                    }
                })
            
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": image_base64
                }
            })
            
            if mask_path:
                with open(mask_path, 'rb') as f:
                    mask_base64 = base64.b64encode(f.read()).decode('utf-8')
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": mask_base64
                    }
                })
            
            # Resolution logic
            resolution_str = "1K"
            if width > 0 and height > 0:
                if width >= 4096 or height >= 4096: resolution_str = "4K"
                elif width >= 2048 or height >= 2048: resolution_str = "2K"
            
            is_pro = "pro" in self.model.lower() or "gemini-3" in self.model.lower()
            
            def _build_edit_payload(res_str: str = None):
                gen_cfg = {
                    "temperature": 0.7,
                    "maxOutputTokens": 32768,
                    "candidateCount": 1,
                    "responseModalities": ["TEXT", "IMAGE"],
                }
                if res_str:
                    gen_cfg["imageConfig"] = {
                        "imageSize": res_str,
                        # [FIX] Removed aspect_ratio="1:1"
                    }
                return {
                    "contents": [{"parts": parts}],
                    "generationConfig": gen_cfg
                }
            
            payload = _build_edit_payload(resolution_str if is_pro else None)
            
            url = f"{self.base_url}/{self.rest_model}:generateContent?key={self.api_key}"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Client': 'python-blender-addon',
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            
            if response.status_code == 400 and ("imageSize" in response.text or "imageConfig" in response.text):
                print(" Model might not support imageSize, retrying without it...")
                payload = _build_edit_payload(None)
                response = requests.post(url, headers=headers, json=payload, timeout=300)
            
            if response.status_code != 200:
                raise GeminiAPIError(f"Edit request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            if 'candidates' not in result or not result['candidates']:
                raise GeminiAPIError("No candidates in edit response")
            
            parts = result['candidates'][0]['content']['parts']
            for part in parts:
                inline_data_key = 'inline_data' if 'inline_data' in part else 'inlineData' if 'inlineData' in part else None
                if inline_data_key:
                    inline_data = part[inline_data_key]
                    data_key = 'data' if 'data' in inline_data else 'bytes' if 'bytes' in inline_data else None
                    if data_key and inline_data[data_key]:
                        image_data = base64.b64decode(inline_data[data_key])
                        mime_type = inline_data.get('mime_type', 'image/png')
                        return image_data, mime_type
            
            raise GeminiAPIError("No image found in edit response")
            
        except requests.RequestException as e:
            raise GeminiAPIError(f"Network error during edit: {str(e)}")
        except Exception as e:
            if isinstance(e, GeminiAPIError):
                raise
            raise GeminiAPIError(f"Edit failed: {str(e)}")

def get_api_key() -> Optional[str]:
    """Get API key from environment variable or addon preferences"""
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key:
        return api_key
    
    import bpy
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        if hasattr(prefs, 'api_key') and prefs.api_key.strip():
            return prefs.api_key.strip()
    except:
        pass
    
    try:
        if hasattr(bpy.context.scene, 'gemini_render') and bpy.context.scene.gemini_render.api_key.strip():
            return bpy.context.scene.gemini_render.api_key.strip()
    except:
        pass
    
    return None