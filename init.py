##### python-based Blender library of methods for Belnder as a class that runs with python script

import bpy
import numpy as np
import os


# a function to save the current rendering of Blender object
def save_image(x_res,y_res,path,name):
	bpy.context.scene.render.filepath = os.path.join(path,name+'.jpg')
	bpy.context.scene.render.resolution_x = 2* x_res #perhaps set resolution in code
	bpy.context.scene.render.resolution_y = 2* y_res
	bpy.ops.render.render(write_still=True)

## save the current blend file associated with the class 
def save_file():
	bpy.ops.wm.save_mainfile()

def change_position(obj_name="Cube",new_pos=(0,0,0) ):
	bpy.data.objects[obj_name].location = new_pos


## construct a cube in blender with vertices verts, and faces and place the object in position pos and color cols
def construct_cube(verts,faces,pos=(0,0,0),colors=(0.05, 0.05, 0.8)):
	bpy.context.scene.cursor_location = (0,0,0)
	mymesh = bpy.data.meshes.new("Cube")
	myobject  = bpy.data.objects.new("Cube",mymesh)
	myobject.location = pos
	bpy.context.scene.objects.link(myobject)
	#activeObject = bpy.context.active_object #Set active object to variable
	mymat = bpy.data.materials.new(name="ObjectMat") #set new material to variable
	myobject.data.materials.append(mymat) #add the material to the object
	mymat.diffuse_color = colors #change color
	mymesh.from_pydata(verts,[],faces)
	mymesh.update(calc_edges=True)

def construct_lamp(lamp_type="SUN",pos=(21.03, -8.49, 17.39),rotation=(6.873, 0.744, 0.295)):
	bpy.context.scene.cursor_location = (0,0,0)
	scene = bpy.context.scene
	lamp_data = bpy.data.lamps.new(name="New Lamp", type=lamp_type)
	# Create new object with our lamp datablock
	lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
	# Link lamp object to the scene so it'll appear in this scene
	scene.objects.link(lamp_object)
	lamp_object.location = pos
	lamp_object.rotation_euler =  rotation



def scale_object(obj_name="Cube",scale=(1,1,1)):
	bpy.data.objects[obj_name].scale[0]  *= scale[0]
	bpy.data.objects[obj_name].scale[1]  *= scale[1]
	bpy.data.objects[obj_name].scale[2]  *= scale[2]

def color_object(obj_name="Cube",colors=(0.05,0.05,0.8)):
	bpy.data.objects[obj_name].active_material.diffuse_color = colors

def color_material(mat_name="CAR PAINT",colors=(0.05,0.05,0.8,1)):
	bpy.data.materials[mat_name].diffuse_ramp.elements[6].color = colors

def rotate_object(obj_name="Cube",rotation=(0,0,0)):
	bpy.data.objects[obj_name].rotation_euler = rotation


# a function that takes 7D vector ( 3 RGBcolors , 2 xy position , 1 z rotations ) and perform that .. all the input is between -1,1
def basic_experiment(obj_name="Cube",vec=[-0.95,-0.95,0.8,0,0,0]):
	rotate_object(obj_name,(0,0,180*vec[5]))
	# scale_object(obj_name=obj_name,scale=(1.4,1.4,1.4))
	change_position(obj_name,(7*vec[3],7*vec[4],0))
	color_object(obj_name,((vec[0]+1)/2,(vec[1]+1)/2,(vec[2]+1)/2))

def city_experiment(obj_name="Cube.006",vec=[-0.95,-0.95,0.8,0,0,0]):
	bpy.context.scene.cursor_location = bpy.data.objects[obj_name].location
	# rotate_object(obj_name,(0,0,180*vec[5]))
	# scale_object(obj_name=obj_name,scale=(1.4,1.4,1.4))
	change_position("Camera",(0,0,0))
	change_position("Camera.002",(80*vec[0],80*vec[1],3+(1+vec[2])*10))
	color_material("CAR PAINT",((vec[3]+1)/8,(vec[4]+1)/8,(vec[5]+1)/8,1))
