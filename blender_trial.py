import bpy
import numpy as np

bpy.context.scene.cursor_location = (0,0,0)
verts = [(0,0,0),(0,5,0),(5,5,0),(5,0,0),(0,0,5),(0,5,5),(5,5,5),(5,0,5)]
faces = [(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
base_verts = 


mymesh = bpy.data.meshes.new("Cube")
myobject  = bpy.data.objects.new("Cube",mymesh)
myobject.location = bpy.context.scene.cursor_location
base_mesh = bpy.data.meshes.new("Plane")
base_object  = bpy.data.objects.new("Plane",base_mesh)


bpy.context.scene.objects.link(myobject)
bpy.context.scene.objects.link(base_object)
#activeObject = bpy.context.active_object #Set active object to variable
mymat = bpy.data.materials.new(name="ObjectMat") #set new material to variable
myobject.data.materials.append(mymat) #add the material to the object
mymat.diffuse_color = (0.05, 0.05, 0.8) #change color

base_mat = bpy.data.materials.new(name="BaseMat") #set new material to variable
base_object.data.materials.append(mymat) #add the material to the object
base_mat.diffuse_color = (0.05, 0.05, 0.05) #change color


mymesh.from_pydata(verts,[],faces)
mymesh.update(calc_edges=True)


