##### python-based Blender library of methods for Belnder as a class that runs with python script

import bpy
import numpy as np
import os

X_MIN = -1
X_MAX = 1 

# a function to save the current rendering of Blender object
def save_image(x_res,y_res,path,name):
    bpy.context.scene.render.filepath = os.path.join(path,name+'.jpg')
    bpy.context.scene.render.resolution_x = x_res #perhaps set resolution in code
    bpy.context.scene.render.resolution_y =  y_res
    bpy.ops.render.render(write_still=True)

def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def connect_objects(parent_name="Empty",child_name="Cube"):
    bpy.data.objects[child_name].parent = bpy.data.objects[parent_name]

def import_batch_off(path_to_off_dir='/media/hamdiaj/D/mywork/sublime/vgd/3d/mydee/'):
    file_list = sorted(os.listdir(path_to_off_dir))
    obj_list = [item for item in file_list if item.endswith('.off')]
    for item in obj_list:
        path_to_file = os.path.join(path_to_off_dir, item)
        bpy.ops.import_mesh.off(filepath=path_to_file)


def import_batch_obj(path_to_obj_dir='/media/hamdiaj/D/mywork/sublime/vgd/3d/PASCAL3D+_release1.1/CAD/car/'):
    file_list = sorted(os.listdir(path_to_obj_dir))
    obj_list = [item for item in file_list if item.endswith('.obj')]
    for item in obj_list:
        path_to_file = os.path.join(path_to_obj_dir, item)
        bpy.ops.import_scene.obj(filepath=path_to_file)


def import_off_dataset_to_objects(path_to_dataset_dir=None,ndataset_name=None,ormilzation_dict=None,material_used=None):
    dirs_list = sorted(os.listdir(path_to_dataset_dir))
    dirs_list = [item for item in dirs_list if os.path.isdir(os.path.join(path_to_dataset_dir,item))]
    print(dirs_list)
    construct_empty(obj_name=ndataset_name)    
    for group in dirs_list:
        construct_empty(obj_name=group)
        connect_objects(parent_name=ndataset_name,child_name=group)    
        path_to_off_dir = os.path.join(path_to_dataset_dir,group)
        file_list = sorted(os.listdir(path_to_off_dir))
        obj_list = [item for item in file_list if item.endswith('.off')]
    # loop through the strings in obj_list and add the files to the scene
        for item in obj_list:
            path_to_file = os.path.join(path_to_off_dir, item)
            bpy.ops.import_mesh.off(filepath=path_to_file)
            item_name = bpy.context.selected_objects[0].name
            if material_used is not None:
                bpy.context.selected_objects[0].data.materials.append(bpy.data.materials[material_used])
            connect_objects(parent_name=group,child_name=item_name)
        if ormilzation_dict is not None:
            scaleing = ormilzation_dict[group]
            scale_object(obj_name=group,scale=(scaleing,scaleing,scaleing))




## save the current blend file associated with the class 
def save_file():
    bpy.ops.wm.save_mainfile()


def change_position(obj_name="Cube",new_pos=(0,0,0) ):
    bpy.data.objects[obj_name].location = new_pos


def construct_empty(obj_name='Empty',pos=(0,0,0)):
    bpy.context.scene.cursor_location = (0,0,0)
    myobject  = bpy.data.objects.new(obj_name,None)
    myobject.location = pos
    myobject.empty_draw_type = 'PLAIN_AXES'
    bpy.context.scene.objects.link(myobject)


## construct a cube in blender with vertices verts, and faces and place the object in position pos and color cols
def construct_cube(obj_name="Cube",verts,faces,pos=(0,0,0),colors=(0.05, 0.05, 0.8)):
    bpy.context.scene.cursor_location = (0,0,0)
    mymesh = bpy.data.meshes.new("Cube")
    myobject  = bpy.data.objects.new(obj_name,mymesh)
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

def energize_lamp(lamp_name="Lamp",energy=0):
    bpy.data.lamps[lamp_name].energy= energy

def hide_tree(parent_name="Empty",hide=True):
    bpy.data.objects[parent_name].hide = hide
    bpy.data.objects[parent_name].hide_render = hide
    if bpy.data.objects[parent_name].children:
        for obj in bpy.data.objects[parent_name].children:
            obj.hide_render = hide
            obj.hide = hide
            if obj.children:
                hide_tree(parent_name=obj.name,hide=hide)




def deactivate_all_textures(material_name="Material"):
    for ii in range(len(bpy.data.materials[material_name].use_textures)):
        bpy.data.materials[material_name].use_textures[ii] = False


def activate_texture(material_name="Material",texture_index=0):
    bpy.data.materials[material_name].use_textures[texture_index] = True


def scale_object(obj_name="Cube",scale=(1,1,1)):
    bpy.data.objects[obj_name].scale[0]  *= scale[0]
    bpy.data.objects[obj_name].scale[1]  *= scale[1]
    bpy.data.objects[obj_name].scale[2]  *= scale[2]

def get_nb_children(parent_name="Empty"):
    return len(bpy.data.objects[parent_name].children)

def get_random_children(parent_name="Empty"):
    random_index = np.random.randint(0,get_nb_children(parent_name=parent_name))
    return bpy.data.objects[parent_name].children[random_index]

def color_object(obj_name="Cube",colors=(0.05,0.05,0.8)):
    bpy.data.objects[obj_name].active_material.diffuse_color = colors

def color_material(mat_name="CAR PAINT",colors=(0.05,0.05,0.8,1)):
    bpy.data.materials[mat_name].diffuse_ramp.elements[1].color = colors

def rotate_object(obj_name="Cube",rotation=(0,0,0)):
    bpy.data.objects[obj_name].rotation_euler = rotation

def delta_rotate_object(obj_name="Cube",rotation=(0,0,0)):
    bpy.data.objects[obj_name].delta_rotation_euler = rotation


# a function that takes 7D vector ( 3 RGBcolors , 2 xy position , 1 z rotations ) and perform that .. all the input is between -1,1
def basic_experiment(obj_name="Cube",vec=[-0.95,-0.95,0.8,0,0,0]):
    rotate_object(obj_name,(0,0,180*vec[5]))
    # scale_object(obj_name=obj_name,scale=(1.4,1.4,1.4))
    change_position(obj_name,(7*vec[3],7*vec[4],0))
    color_object(obj_name,((vec[0]+1)/2,(vec[1]+1)/2,(vec[2]+1)/2))

# function that takes 8D vector ( camera distance to object  , 2 Camera azimuth and elevation (-180,180),(0,50)   ,1 light azimth with respect to the camera(-180,180) , 1 light elevation (0,90),
 # 1 3 RGB color of object )  and perform that .. all the input is between X_MIN,X_MAX
def city_experiment(obj_name="myorigin",vec=[0.,0.,0.,0.,0.,0.,0.,0.,0.],parent_name='car'):
    texture_dict={'aeroplane':0,'bench':0 , 'bicycle':0, 'boat':3, 'bottle':1, 'bus':0, 'car':0,'chair':1,'diningtable':1, 'motorbike':0, 'sofa':1, 'train':0, 'tvmonitor':1, 'truck':0}
    bpy.context.scene.cursor_location = (0,0,0)
    for any_parent in texture_dict.keys():
        hide_tree(parent_name=any_parent,hide=True)
    object_instance = get_random_children(parent_name)
    hide_tree(parent_name=object_instance.name,hide=False)
    change_position("Camera",(translate(vec[0],X_MIN,X_MAX,-14.5,-8),-0.35,0.1))
    rotate_object(obj_name,(0,translate(vec[2],X_MIN,X_MAX,0,0.9),translate(vec[1],X_MIN,X_MAX,-3.15,3.15)))
    rotate_object("nextorigin",(0,translate(vec[4],X_MIN,X_MAX,0.05,1.57),translate(vec[3],X_MIN,X_MAX,-3.15,3.15)))
    # energize_lamp(lamp_name="Lamp.002",energy=translate(vec[5],X_MIN,X_MAX,0.3,2.5))
    color_material("CAR PAINT",(translate(vec[5],X_MIN,X_MAX,0,1),translate(vec[6],X_MIN,X_MAX,0,1),translate(vec[7],X_MIN,X_MAX,0,1),1))
    deactivate_all_textures('material_1.001')
    activate_texture('material_1.001',texture_dict[parent_name])

def prepare_dataset():
    # normalization_dict={'aeroplane':4.6, 'bicycle':3, 'boat':4.8, 'bottle':2, 'bus':5, 'car':4,'chair':1.7, 'diningtable':2.7, 'motorbike':3, 'sofa':3, 'train':5, 'tvmonitor':2}
    path_to_dataset_dir = '/media/hamdiaj/D/mywork/sublime/vgd/3d/mydee/'
    import_off_dataset_to_objects(path_to_dataset_dir=path_to_dataset_dir,ndataset_name='mydataset',material_used='CAR PAINT')