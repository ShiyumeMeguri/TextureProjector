"""
Projection core for TextureProjector.

Architecture (v2): every capture goes through a real Blender camera render.
When the user is not looking through a camera, a temporary "view camera" is
built from the viewport matrices. The color capture, the depth capture and
the UV projection all derive from the SAME camera + resolution, so the
AI image and the mesh UVs are aligned by construction — this removes the
old viewport-alignment known issue entirely.

Performance: UV projection and face/material assignment are fully numpy
vectorized (foreach_get/foreach_set), no per-vertex Python loops.
Armature/modifier deformation is respected by projecting the evaluated
(depsgraph) vertex positions when topology allows — fixing the old
"posed armature misaligns projection" known issue.
"""

import math
import os
import tempfile

import numpy as np
import bpy
import bmesh

from . import depth_utils

PROJECTED_UV_NAME = "Projected UVs"
IMAGE_NODE_NAME = "Gemini_Image_Node"

# Official NanoBanana output resolutions per aspect ratio (pixel-exact).
SUPPORTED_RESOLUTIONS = {
    "1:1":   (1024, 1024),
    "2:3":   (832, 1248),
    "3:2":   (1248, 832),
    "3:4":   (864, 1184),
    "4:3":   (1184, 864),
    "4:5":   (896, 1152),
    "5:4":   (1152, 896),
    "9:16":  (768, 1344),
    "16:9":  (1344, 768),
    "21:9":  (1536, 672),
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _closest_supported(res_x: int, res_y: int):
    if res_y == 0:
        return "1:1"
    target = res_x / res_y
    return min(SUPPORTED_RESOLUTIONS.items(),
               key=lambda kv: abs(target - kv[1][0] / kv[1][1]))[0]


def snap_to_supported_resolution(res_x: int, res_y: int):
    """Snap to the closest supported ratio, preserving approximate scale."""
    key = _closest_supported(res_x, res_y)
    base_w, base_h = SUPPORTED_RESOLUTIONS[key]
    scale = max(res_x / base_w, res_y / base_h, 1.0)
    w = int(base_w * scale)
    h = int(base_h * scale)
    return (w + w % 2, h + h % 2)


def get_capture_dimensions(scene, region, rv3d):
    """Return (width, height, use_scene_camera) for this capture.

    Camera view  -> snapped scene render resolution (user-controlled scale).
    Viewport view -> the exact official base resolution closest to the
                     region aspect (cheapest pixel-perfect input).
    """
    is_camera_view = bool(rv3d) and rv3d.view_perspective == 'CAMERA'
    use_scene_camera = is_camera_view and scene.camera is not None

    if use_scene_camera:
        w, h = snap_to_supported_resolution(
            scene.render.resolution_x, scene.render.resolution_y)
        print(f"[GEMINI] Camera capture: {scene.render.resolution_x}x"
              f"{scene.render.resolution_y} -> {w}x{h}")
    else:
        key = _closest_supported(region.width, region.height)
        w, h = SUPPORTED_RESOLUTIONS[key]
        print(f"[GEMINI] Viewport capture: region {region.width}x{region.height}"
              f" -> {key} ({w}x{h})")
    return w, h, use_scene_camera


def setup_temporary_workspace(props) -> str:
    """Create the working directory (persistent next to the .blend in debug)."""
    if props.debug_mode:
        blend_path = bpy.data.filepath
        base = (os.path.join(os.path.dirname(blend_path), "textures")
                if blend_path else os.path.join(tempfile.gettempdir(), "textures"))
        temp_dir = os.path.join(base, "gemini_debug_session")
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="gemini_proj_")
    return temp_dir


# ---------------------------------------------------------------------------
# View camera (the fix for viewport pixel alignment)
# ---------------------------------------------------------------------------

def create_view_camera(scene, space, rv3d, width: int, height: int):
    """Build a temporary camera that reproduces the current viewport view.

    The camera FOV is derived from the viewport window matrix, so a render
    through this camera matches what the UV projection computes exactly.
    """
    cam_data = bpy.data.cameras.new("Gemini_ViewCam")
    cam_obj = bpy.data.objects.new("Gemini_ViewCam", cam_data)
    scene.collection.objects.link(cam_obj)

    cam_obj.matrix_world = rv3d.view_matrix.inverted()

    P = rv3d.window_matrix
    horizontal = width >= height
    cam_data.sensor_fit = 'HORIZONTAL' if horizontal else 'VERTICAL'

    if rv3d.view_perspective == 'ORTHO':
        cam_data.type = 'ORTHO'
        # P[0][0] = 2 / world_width, P[1][1] = 2 / world_height
        span = 2.0 / (P[0][0] if horizontal else P[1][1])
        cam_data.ortho_scale = abs(span)
    else:
        cam_data.type = 'PERSP'
        diag = P[0][0] if horizontal else P[1][1]
        cam_data.angle = 2.0 * math.atan(1.0 / abs(diag))

    if space is not None:
        cam_data.clip_start = space.clip_start
        cam_data.clip_end = space.clip_end
    return cam_obj


def resolve_capture_camera(context, scene, space, rv3d, width, height,
                           use_scene_camera):
    """Return (camera_object, is_temporary)."""
    if use_scene_camera:
        return scene.camera, False
    return create_view_camera(scene, space, rv3d, width, height), True


def camera_matrices(context, cam_obj, width, height):
    """View + projection matrices shared by depth capture and UV projection."""
    depsgraph = context.evaluated_depsgraph_get()
    view = cam_obj.matrix_world.inverted()
    proj = cam_obj.calc_matrix_camera(depsgraph, x=width, y=height)
    return view, proj


# ---------------------------------------------------------------------------
# Captures (always through a camera render — no OpenGL viewport path)
# ---------------------------------------------------------------------------

def _sync_viewport_visibility(context):
    """Hide render-visible objects that the user hid in the viewport (WYSIWYG)."""
    hidden = []
    try:
        for obj in context.view_layer.objects:
            if not obj.visible_get() and not obj.hide_render:
                obj.hide_render = True
                hidden.append(obj)
    except Exception as e:
        print(f"[GEMINI] Visibility sync warning: {e}")
    return hidden


def capture_color(context, scene, cam_obj, space, width, height,
                  filepath) -> bool:
    """Render the scene through cam_obj to filepath.

    Assumes scene resolution is already set to (width, height) and the
    output format to PNG (see render_state_guard in operators).
    """
    saved = {
        'camera': scene.camera,
        'engine': scene.render.engine,
        'filepath': scene.render.filepath,
        'display_light': scene.display.shading.light,
        'display_color': scene.display.shading.color_type,
    }
    saved_display_type = getattr(scene.display.shading, 'type', None)
    hidden = _sync_viewport_visibility(context)

    try:
        scene.camera = cam_obj
        scene.render.filepath = filepath

        view_shading = space.shading.type if space else 'SOLID'
        if view_shading in {'MATERIAL', 'RENDERED'}:
            try:
                scene.render.engine = 'BLENDER_EEVEE_NEXT'
            except TypeError:
                scene.render.engine = 'BLENDER_EEVEE'
        else:
            scene.render.engine = 'BLENDER_WORKBENCH'
            if saved_display_type is not None:
                try:
                    scene.display.shading.type = 'SOLID'
                except TypeError:
                    pass
            if space:
                scene.display.shading.color_type = space.shading.color_type
                scene.display.shading.light = space.shading.light

        bpy.ops.render.render(write_still=True)
        ok = os.path.exists(filepath)
        if ok:
            print(f"[GEMINI] Capture saved: {filepath}")
        return ok
    except Exception as e:
        print(f"[GEMINI] Color capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for obj in hidden:
            try:
                obj.hide_render = False
            except ReferenceError:
                pass
        scene.render.engine = saved['engine']
        scene.render.filepath = saved['filepath']
        scene.camera = saved['camera']
        if saved_display_type is not None:
            try:
                scene.display.shading.type = saved_display_type
            except TypeError:
                pass
        scene.display.shading.color_type = saved['display_color']
        scene.display.shading.light = saved['display_light']


def capture_depth(context, cam_obj, width, height, filepath) -> bool:
    try:
        view, proj = camera_matrices(context, cam_obj, width, height)
        depth_utils.render_depth_map(context, view, proj, width, height,
                                     filepath, invert=True)
        return os.path.exists(filepath)
    except Exception as e:
        print(f"[GEMINI] Depth capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_wireframe(context, scene, cam_obj, area, region, space, rv3d,
                      width, height, filepath) -> bool:
    """Render Blender's own viewport wireframe through the capture camera.

    The F12 render engines have no wireframe mode, so this drives the
    viewport draw engine instead: temporarily look through the capture
    camera and take an OpenGL viewport render with WIREFRAME shading.
    In camera view the OpenGL render outputs the camera frame at the
    scene render resolution, so it stays pixel-aligned with the color
    and depth captures (which use the same camera).
    """
    saved = {
        'camera': scene.camera,
        'filepath': scene.render.filepath,
        'view_perspective': rv3d.view_perspective,
        'shading_type': space.shading.type,
        'overlays': space.overlay.show_overlays,
    }
    try:
        scene.camera = cam_obj
        scene.render.filepath = filepath
        rv3d.view_perspective = 'CAMERA'
        space.shading.type = 'WIREFRAME'
        space.overlay.show_overlays = False

        with context.temp_override(window=context.window, area=area,
                                   region=region, scene=scene,
                                   space_data=space):
            bpy.ops.render.opengl(write_still=True, view_context=True)

        ok = os.path.exists(filepath)
        if ok:
            print(f"[GEMINI] Wireframe capture saved: {filepath}")
        return ok
    except Exception as e:
        print(f"[GEMINI] Wireframe capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        scene.camera = saved['camera']
        scene.render.filepath = saved['filepath']
        try:
            space.shading.type = saved['shading_type']
        except TypeError:
            pass
        space.overlay.show_overlays = saved['overlays']
        rv3d.view_perspective = saved['view_perspective']


# ---------------------------------------------------------------------------
# UV projection (numpy)
# ---------------------------------------------------------------------------

def _face_mask_to_loop_mask(mesh, face_mask: np.ndarray) -> np.ndarray:
    n_polys = len(mesh.polygons)
    loop_total = np.empty(n_polys, dtype=np.int32)
    mesh.polygons.foreach_get('loop_total', loop_total)
    return np.repeat(face_mask, loop_total)


def _evaluated_coords(obj, depsgraph, n_verts: int):
    """Deformed vertex coordinates if topology matches, else rest coords.

    Using the evaluated mesh makes projection follow armature poses and
    deform modifiers, matching what the render captured.
    """
    coords = None
    try:
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        if eval_mesh is not None:
            if len(eval_mesh.vertices) == n_verts:
                coords = np.empty(n_verts * 3, dtype=np.float64)
                eval_mesh.vertices.foreach_get('co', coords)
            eval_obj.to_mesh_clear()
    except Exception as e:
        print(f"[GEMINI] Evaluated mesh unavailable for {obj.name}: {e}")

    if coords is None:
        coords = np.empty(n_verts * 3, dtype=np.float64)
        obj.data.vertices.foreach_get('co', coords)
    return coords.reshape(-1, 3)


def project_uvs(obj, context, cam_obj, width, height,
                face_mask: np.ndarray = None,
                uv_name: str = PROJECTED_UV_NAME) -> bool:
    """Write camera-space UVs for the masked faces of obj (OBJECT mode).

    Uses exactly the same matrices as the captures for pixel alignment.
    """
    mesh = obj.data
    n_verts = len(mesh.vertices)
    n_loops = len(mesh.loops)
    n_polys = len(mesh.polygons)
    if n_verts == 0 or n_loops == 0 or n_polys == 0:
        return False

    if face_mask is None:
        face_mask = np.ones(n_polys, dtype=bool)
    if not face_mask.any():
        return False

    depsgraph = context.evaluated_depsgraph_get()
    coords = _evaluated_coords(obj, depsgraph, n_verts)

    view, proj = camera_matrices(context, cam_obj, width, height)
    mvp = np.array(proj @ view @ obj.matrix_world, dtype=np.float64)

    homo = np.empty((n_verts, 4), dtype=np.float64)
    homo[:, :3] = coords
    homo[:, 3] = 1.0
    clip = homo @ mvp.T
    w = clip[:, 3].copy()
    w[np.abs(w) < 1e-9] = 1e-9
    uvs = clip[:, :2] / w[:, None] * 0.5 + 0.5  # NDC -> [0, 1]

    loop_verts = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', loop_verts)

    uv_layer = mesh.uv_layers.get(uv_name) or mesh.uv_layers.new(name=uv_name)
    if uv_layer is None:
        # Meshes are capped at 8 UV layers; surface a clear error instead
        # of crashing on a None layer.
        raise RuntimeError(
            f"'{obj.name}' has no free UV layer slot for '{uv_name}' "
            f"(Blender allows at most 8 UV maps per mesh)")
    buf = np.empty(n_loops * 2, dtype=np.float32)
    uv_layer.data.foreach_get('uv', buf)
    buf = buf.reshape(-1, 2)

    loop_mask = _face_mask_to_loop_mask(mesh, face_mask)
    buf[loop_mask] = uvs[loop_verts[loop_mask]]
    uv_layer.data.foreach_set('uv', buf.ravel())
    mesh.update()
    return True


def resolve_dest_uv(obj, requested: str) -> str:
    """Resolve the bake-destination UV map name for an object.

    Empty request = the object's first UV map (UV0). A named UV map that
    does not exist yet is created so the bake can target it.
    """
    uvs = obj.data.uv_layers
    requested = (requested or "").strip()
    if requested:
        if requested in uvs:
            return requested
        new_layer = uvs.new(name=requested)
        if new_layer is not None:
            return new_layer.name
        print(f"[GEMINI] Could not create UV map '{requested}' on {obj.name} "
              f"(8-layer limit?), falling back to UV0")
    return uvs[0].name if len(uvs) else ""


def rollback_projection_materials(registry):
    """Undo the material mutations of a failed/cancelled projection.

    Without this, an API error would leave the objects wearing the empty
    projection material — rendering them black. Restores the recorded
    per-face material indices and removes the slot we appended.
    """
    for entry in registry.get('material_rollback', []):
        obj = bpy.data.objects.get(entry['object_name'])
        if not obj or obj.type != 'MESH':
            continue
        mesh = obj.data
        try:
            if entry.get('added_slot'):
                # Our slot was appended last, so popping it does not shift
                # the indices of any pre-existing slot.
                for i in range(len(mesh.materials) - 1, -1, -1):
                    mat = mesh.materials[i]
                    if mat and mat.name == entry['material_name']:
                        mesh.materials.pop(index=i)
                        break
            indices = entry['indices']
            if len(mesh.polygons) == len(indices):
                mesh.polygons.foreach_set('material_index', indices)
            mesh.update()

            mat = bpy.data.materials.get(entry['material_name'])
            if mat and mat.users == 0:
                bpy.data.materials.remove(mat)
            print(f"[GEMINI] Rolled back projection material on {obj.name}")
        except Exception as e:
            print(f"[GEMINI] Material rollback error on {entry['object_name']}: {e}")
    registry['material_rollback'] = []


def assign_material_to_faces(obj, material, face_mask: np.ndarray = None) -> int:
    """Ensure material has a slot on obj and assign it to masked faces."""
    mesh = obj.data
    slot_index = -1
    for i, slot in enumerate(obj.material_slots):
        if slot.material == material:
            slot_index = i
            break
    if slot_index == -1:
        mesh.materials.append(material)
        slot_index = len(mesh.materials) - 1

    n_polys = len(mesh.polygons)
    if n_polys:
        indices = np.empty(n_polys, dtype=np.int32)
        mesh.polygons.foreach_get('material_index', indices)
        if face_mask is None:
            indices[:] = slot_index
        else:
            indices[face_mask] = slot_index
        mesh.polygons.foreach_set('material_index', indices)
        mesh.update()
    return slot_index


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def setup_projection_material(material_name="Gemini_Projection_Material",
                              use_image=None, emission_color=None):
    """Create/reset an emission projection material.

    CRITICAL: each mesh MUST get its own material instance (unique name per
    object). Sharing one material across a multi-selection overwrites the
    previous object's projection. Strictly keep the 1:1 object-to-material
    mapping.
    """
    material = bpy.data.materials.get(material_name)
    if not material:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (400, 0)
    emit_node = nodes.new("ShaderNodeEmission")
    emit_node.location = (200, 0)
    emit_node.inputs['Strength'].default_value = 1.0
    links.new(emit_node.outputs['Emission'], output_node.inputs['Surface'])

    image_node = None
    uv_map_node = None
    if emission_color:
        emit_node.inputs['Color'].default_value = emission_color
        material.diffuse_color = emission_color  # visible in Solid shading too
    else:
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.name = IMAGE_NODE_NAME
        image_node.location = (0, 0)
        if use_image:
            image_node.image = use_image

        uv_map_node = nodes.new("ShaderNodeUVMap")
        uv_map_node.name = "Gemini_UV_Map"
        uv_map_node.uv_map = PROJECTED_UV_NAME
        uv_map_node.location = (-200, 0)

        links.new(uv_map_node.outputs['UV'], image_node.inputs['Vector'])
        links.new(image_node.outputs['Color'], emit_node.inputs['Color'])

    return material, image_node, uv_map_node


def find_object_texture(obj):
    """First image texture found in the object's materials (repair target)."""
    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    return node.image
    return None


# ---------------------------------------------------------------------------
# Mask repair meshes (OBJECT-mode, numpy selection driven)
# ---------------------------------------------------------------------------

def setup_mask_repair_meshes(context, props, registry, face_masks):
    """Duplicate selected faces into overlay meshes with a mask material.

    face_masks: {object_name: bool ndarray per polygon}.
    Registers all temp datablocks in the registry for guaranteed cleanup.
    """
    mask_repair_data = {
        'original_textures': {},
        'original_object_names': [],
        'temp_to_original': {},
    }

    for obj_name, face_mask in face_masks.items():
        obj = bpy.data.objects.get(obj_name)
        if not obj or obj.type != 'MESH' or not face_mask.any():
            continue

        original_texture = find_object_texture(obj)
        if not original_texture:
            print(f"[GEMINI] Mask repair: '{obj_name}' has no texture, skipped")
            continue

        # Build a mesh containing only the selected faces.
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        doomed = [f for f in bm.faces if not face_mask[f.index]]
        bmesh.ops.delete(bm, geom=doomed, context='FACES')

        temp_mesh = bpy.data.meshes.new(f"{obj_name}_MaskTemp_Data")
        bm.to_mesh(temp_mesh)
        bm.free()

        temp_obj = bpy.data.objects.new(f"{obj_name}_MaskTemp", temp_mesh)
        temp_obj.matrix_world = obj.matrix_world.copy()
        context.collection.objects.link(temp_obj)
        registry['temp_objects'].append(temp_obj.name)

        # Anti z-fighting shell.
        solidify = temp_obj.modifiers.new(name="Gemini_Solidify", type='SOLIDIFY')
        solidify.thickness = 0.002
        solidify.offset = 0
        solidify.use_rim = True

        mask_mat_name = f"Gemini_Mask_{obj_name}_Material_Temp"
        old = bpy.data.materials.get(mask_mat_name)
        if old:
            bpy.data.materials.remove(old)
        color = (props.mask_color[0], props.mask_color[1], props.mask_color[2], 1.0)
        mask_mat, _, _ = setup_projection_material(
            material_name=mask_mat_name, emission_color=color)
        registry['temp_materials'].append(mask_mat_name)

        temp_mesh.materials.clear()
        temp_mesh.materials.append(mask_mat)

        mask_repair_data['original_textures'][obj_name] = original_texture.name
        mask_repair_data['original_object_names'].append(obj_name)
        mask_repair_data['temp_to_original'][temp_obj.name] = obj_name

    if not mask_repair_data['original_object_names']:
        return None
    print(f"[GEMINI] Mask repair: {len(mask_repair_data['original_object_names'])}"
          " overlay meshes created")
    return mask_repair_data


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------

def bake(context, obj, texture_node_name, target_image,
         src_uv_name=PROJECTED_UV_NAME, margin=16, use_clear=True):
    """Bake the projected emission into target_image (Cycles EMIT, 1 sample).

    Saves and restores EVERY setting it touches — including Cycles sample
    counts, which the previous version leaked (leaving user scenes at
    1 sample after a bake).
    """
    scene = context.scene

    saved = {
        'engine': scene.render.engine,
        'active': context.view_layer.objects.active,
        'mode': obj.mode,
        'hide_render': obj.hide_render,
        'dither': scene.render.dither_intensity,
        'view_transform': scene.view_settings.view_transform,
        'bake_clear': scene.render.bake.use_clear,
        'bake_margin': scene.render.bake.margin,
        'bake_target': scene.render.bake.target,
        'bake_s2a': scene.render.bake.use_selected_to_active,
        'selection': [o.name for o in context.selected_objects],
    }
    saved_cycles = {}

    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    obj.hide_render = False

    mats_data = []
    try:
        scene.render.engine = 'CYCLES'
        cycles = getattr(scene, 'cycles', None)
        if cycles is not None:
            for attr, value in (('samples', 1),
                                ('use_adaptive_sampling', False),
                                ('use_denoising', False)):
                if hasattr(cycles, attr):
                    saved_cycles[attr] = getattr(cycles, attr)
                    setattr(cycles, attr, value)
            if hasattr(cycles, 'device'):
                saved_cycles['device'] = cycles.device
                prefs = context.preferences.addons.get('cycles')
                has_gpu = bool(prefs) and prefs.preferences.compute_device_type != 'NONE'
                cycles.device = 'GPU' if has_gpu else 'CPU'

        scene.render.dither_intensity = 0.0
        scene.view_settings.view_transform = 'Standard'

        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # Wire every material for a pure color transfer (emission bypass).
        unique_materials = {slot.material for slot in obj.material_slots
                            if slot.material and slot.material.use_nodes}
        for mat in unique_materials:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            target_node = nodes.get("Gemini_Bake_Target") or nodes.new('ShaderNodeTexImage')
            target_node.name = "Gemini_Bake_Target"
            target_node.image = target_image
            for n in nodes:
                n.select = False
            target_node.select = True
            nodes.active = target_node

            surface_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            src_node = nodes.get(texture_node_name)
            temp_emit = None
            temp_uv = None
            original_links = []
            original_vector_links = []

            if surface_output and src_node:
                for link in list(src_node.inputs['Vector'].links):
                    original_vector_links.append((link.from_socket, link.to_socket))
                    links.remove(link)
                temp_uv = nodes.new('ShaderNodeUVMap')
                temp_uv.uv_map = src_uv_name
                links.new(temp_uv.outputs['UV'], src_node.inputs['Vector'])

                for link in list(surface_output.inputs['Surface'].links):
                    original_links.append((link.from_socket, link.to_socket))
                    links.remove(link)
                temp_emit = nodes.new('ShaderNodeEmission')
                temp_emit.inputs['Strength'].default_value = 1.0
                links.new(src_node.outputs['Color'], temp_emit.inputs['Color'])
                links.new(temp_emit.outputs['Emission'], surface_output.inputs['Surface'])

            mats_data.append((mat, original_links, temp_emit, target_node,
                              temp_uv, original_vector_links))

        if not mats_data:
            raise RuntimeError(f"Object {obj.name} has no node materials to bake")

        scene.render.bake.use_clear = use_clear
        scene.render.bake.margin = margin
        scene.render.bake.target = 'IMAGE_TEXTURES'
        scene.render.bake.use_selected_to_active = False

        print(f"[GEMINI] Baking {obj.name} -> {target_image.name} "
              f"(margin={margin}, clear={use_clear})")
        bpy.ops.object.bake(type='EMIT')

    finally:
        for mat, original_links, temp_emit, target_node, temp_uv, vec_links in mats_data:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            if temp_emit:
                nodes.remove(temp_emit)
            if temp_uv:
                nodes.remove(temp_uv)
            if target_node and target_node.name in nodes:
                nodes.remove(target_node)
            surface_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if surface_output:
                for from_sock, to_sock in original_links:
                    links.new(from_sock, to_sock)
            src_node = nodes.get(texture_node_name)
            if src_node:
                for from_sock, to_sock in vec_links:
                    links.new(from_sock, to_sock)

        cycles = getattr(scene, 'cycles', None)
        if cycles is not None:
            for attr, value in saved_cycles.items():
                try:
                    setattr(cycles, attr, value)
                except Exception:
                    pass

        scene.render.engine = saved['engine']
        scene.render.dither_intensity = saved['dither']
        scene.view_settings.view_transform = saved['view_transform']
        scene.render.bake.use_clear = saved['bake_clear']
        scene.render.bake.margin = saved['bake_margin']
        scene.render.bake.target = saved['bake_target']
        scene.render.bake.use_selected_to_active = saved['bake_s2a']

        try:
            bpy.ops.object.select_all(action='DESELECT')
            for name in saved['selection']:
                o = bpy.data.objects.get(name)
                if o:
                    o.select_set(True)
        except Exception:
            pass
        if saved['active'] and saved['active'].name in bpy.data.objects:
            context.view_layer.objects.active = saved['active']
        obj.hide_render = saved['hide_render']
        if saved['mode'] != obj.mode and obj == context.view_layer.objects.active:
            try:
                bpy.ops.object.mode_set(mode=saved['mode'])
            except RuntimeError:
                pass


def finalize_object_materials(target_obj, target_img, dest_uv_name,
                              search_img=None, node_name=None):
    """Point the object's texture nodes at the baked image + destination UVs."""
    if not target_obj or not target_obj.data or not hasattr(target_obj.data, 'materials'):
        return

    for slot in target_obj.material_slots:
        mat = slot.material
        if not (mat and mat.use_nodes):
            continue
        m_nodes = mat.node_tree.nodes
        m_links = mat.node_tree.links

        for node in m_nodes:
            if node.type != 'TEX_IMAGE':
                continue
            is_match = (
                (node.image and (node.image == target_img
                                 or (search_img and node.image == search_img)))
                or node.name == node_name
                or (target_img and node.image and node.image.name == target_img.name)
            )
            if not is_match:
                continue

            if target_img:
                node.image = target_img

            uv_node = None
            if node.inputs['Vector'].is_linked:
                from_node = node.inputs['Vector'].links[0].from_node
                if from_node.type == 'UV_MAP':
                    uv_node = from_node
            if not uv_node and dest_uv_name:
                for link in list(node.inputs['Vector'].links):
                    m_links.remove(link)
                uv_node = m_nodes.new('ShaderNodeUVMap')
                m_links.new(uv_node.outputs['UV'], node.inputs['Vector'])
            if uv_node and dest_uv_name:
                uv_node.uv_map = dest_uv_name

    if dest_uv_name and dest_uv_name in target_obj.data.uv_layers:
        target_obj.data.uv_layers.active = target_obj.data.uv_layers[dest_uv_name]
        target_obj.data.uv_layers[dest_uv_name].active_render = True


def perform_projection_bake(context, obj, texture_node_name, target_image,
                            src_uv_name, dest_uv_name, margin=16,
                            use_clear=True, is_mask_repair=False,
                            original_obj=None, search_img=None) -> bool:
    """Unified bake wrapper for both normal and mask-repair modes."""
    try:
        if is_mask_repair:
            margin, use_clear = 1, False

        # Cycles writes the bake through the active-render UV map, so point
        # it at the requested destination before baking.
        if dest_uv_name and dest_uv_name in obj.data.uv_layers:
            obj.data.uv_layers[dest_uv_name].active_render = True

        bake(context=context, obj=obj, texture_node_name=texture_node_name,
             target_image=target_image, src_uv_name=src_uv_name,
             margin=margin, use_clear=use_clear)

        finalize_target = original_obj if (is_mask_repair and original_obj) else obj
        finalize_object_materials(
            target_obj=finalize_target, target_img=target_image,
            dest_uv_name=dest_uv_name, search_img=search_img,
            node_name=texture_node_name)
        return True
    except Exception as e:
        print(f"[GEMINI] Bake failed for {obj.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Validation / target gathering
# ---------------------------------------------------------------------------

def gather_projection_targets(context):
    """Collect target meshes.

    One-click semantics:
      - Object Mode: all faces of every SELECTED mesh are targeted. An
        active-but-unselected mesh is deliberately NOT a fallback — the
        user thinks nothing is selected, and silently kicking off a full
        capture render reads as a UI hang.
      - Edit Mode: the meshes being edited; only their selected faces are
        targeted (mesh data must be synced to OBJECT mode by the caller
        before reading the selection).
    Returns (targets, was_edit_mode).
    """
    was_edit = (context.mode == 'EDIT_MESH')
    if was_edit:
        targets = [o for o in context.objects_in_mode if o.type == 'MESH']
    else:
        targets = [o for o in context.selected_objects
                   if o.type == 'MESH' and '_MaskTemp' not in o.name]
    return targets, was_edit


def build_face_masks(targets, from_selection: bool):
    face_masks = {}
    for obj in targets:
        n = len(obj.data.polygons)
        if n == 0:
            continue
        if from_selection:
            mask = np.empty(n, dtype=bool)
            obj.data.polygons.foreach_get('select', mask)
        else:
            mask = np.ones(n, dtype=bool)
        if mask.any():
            face_masks[obj.name] = mask
    return face_masks


def validate_projection(context):
    """Raise with a user-readable message if projection cannot run."""
    targets, _ = gather_projection_targets(context)
    if not targets:
        raise RuntimeError(
            "Select at least one mesh object (Object Mode projects all faces; "
            "Edit Mode projects the selected faces).")
    return None
