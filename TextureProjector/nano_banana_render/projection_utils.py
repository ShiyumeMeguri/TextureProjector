import bpy
import bmesh
import gpu
import gpu.texture
from gpu_extras.batch import batch_for_shader
import mathutils
import numpy as np

def validate_projection(context):
    """Validate if projection can be performed. Ensures selected meshes are in Edit Mode with selected faces."""
    selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
    
    if not selected_meshes:
        raise Exception("No mesh objects selected. Select at least one mesh to project onto.")
    
    import bmesh
    has_selection = False
    for obj in selected_meshes:
        # Check if object is in Edit Mode
        if obj.mode != 'EDIT':
            continue
            
        bm = bmesh.from_edit_mesh(obj.data)
        if not bm:
            continue
            
        for f in bm.faces:
            if f.select:
                has_selection = True
                break
        if has_selection:
            break
    
    if not has_selection:
        raise Exception("No faces selected in Edit Mode. Please enter Edit Mode for your meshes and select projection faces.")
    return None

def bake(context, mesh, src, dest, src_uv, dest_uv):
    """Bake projected texture to destination texture using GPU (Direct Port from Dream Textures)"""
    def bake_shader():
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC2', "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.sampler(0, 'FLOAT_2D', "image")
        shader_info.vertex_in(0, 'VEC2', "src_uv")
        shader_info.vertex_in(1, 'VEC2', "dest_uv")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "fragColor")

        shader_info.vertex_source("""
void main()
{
    gl_Position = vec4(dest_uv * 2 - 1, 0.0, 1.0);
    uvInterp = src_uv;
}
""")

        shader_info.fragment_source("""
void main()
{
    fragColor = texture(image, uvInterp);
}
""")

        return gpu.shader.create_from_info(shader_info)

    width, height = dest.size[0], dest.size[1]
    offscreen = gpu.types.GPUOffScreen(width, height)

    buffer = gpu.types.Buffer('FLOAT', width * height * 4, src)
    texture = gpu.types.GPUTexture(size=(width, height), data=buffer, format='RGBA16F')
    
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(mathutils.Matrix.Identity(4))

            # indices for triangles - must use calc_loop_triangles on BMesh
            vertices = np.array([[l.vert.index for l in loop] for loop in mesh.calc_loop_triangles()], dtype='i')

            shader = bake_shader()
            batch = batch_for_shader(
                shader, 'TRIS',
                {"src_uv": src_uv, "dest_uv": dest_uv},
                indices=vertices,
            )
            shader.uniform_sampler("image", texture)
            batch.draw(shader)
        
        # Consistent with dream-textures implementation
        projected = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
    
    offscreen.free()
    dest.pixels[:] = projected.ravel()
