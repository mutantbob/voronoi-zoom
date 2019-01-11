import bpy
from math import *
import random



class VoronoiNode:
    def __init__(self, xyz, color):
        self.xyz =xyz
        self.color = color


def rigInputsFor(gr, voronois, nx, ny, coordIn, colorIn, uvSrc, lineSrc):
    if len(voronois)>1:
        y3, x3, node = nodesFor(gr, voronois, nx, ny, uvSrc, lineSrc)
        gr.links.new(coordIn, node.outputs[0])
        gr.links.new(colorIn, node.outputs[1])
        return y3, x3
    else:
        coordIn.default_value = voronois[0].xyz
        colorIn.default_value = voronois[0].color
        return ny+100, nx


def nodesFor(gr, voronois, nx,ny, uvSrc, lineSrc):

    print ("nodesFor(,,,,%r, %r)"%(uvSrc, lineSrc))
    print ("nodesFor(,,,,%r, %r)"%(uvSrc.node, lineSrc.node))

    split = floor(len(voronois)/2)

    v1 = voronois[:split]
    v2 = voronois[split:]

    node = gr.nodes.new('ShaderNodeGroup')
    node.node_tree = voronoiDiscriminator()

    node.location = (nx, ny)

    print(node.inputs[0])
    print(uvSrc)

    try:
        gr.links.new(node.inputs[0], uvSrc)
        gr.links.new(node.inputs[3], lineSrc)
    except BaseException as e:
        print(e)

    y2, x2 = rigInputsFor(gr, v1, nx-200, ny, node.inputs[4], node.inputs[5], uvSrc, lineSrc)
    y3, x3 = rigInputsFor(gr, v2, nx-200, y2, node.inputs[1], node.inputs[2], uvSrc, lineSrc)

    return y3, min(x2,x3), node


def socketGetOrNew(sockets, bl_idname, name, idx):
    if idx+1<len(sockets): # there is a phantom socket at the end of the list
        return sockets[idx]

    print("sockets len before %d"%(len(sockets)))
    rval = sockets.new(bl_idname, name)
    print("%r.new(%s, %s)"%(sockets, bl_idname, name))
    print("sockets len after %d"%(len(sockets)))
    if rval.name != name:
        print("%s != %s"%(rval.name, name))
        #rval.name = name
    return rval

def mission1():

    groupName = "vorono 1"
    gr = bpy.data.node_groups.get(groupName)
    if gr is None:
        gr = bpy.data.node_groups.new(groupName, 'ShaderNodeTree')
    else:
        while len(gr.nodes) > 0:
            gr.nodes.remove(gr.nodes[-1])
    
    print(gr)

    grIn = gr.nodes.new('NodeGroupInput')
    grOut = gr.nodes.new('NodeGroupOutput')

    #node = gr.nodes.new('ShaderNodeGroup')

    uvSrc = socketGetOrNew(grIn.outputs, 'NodeSocketVector', 'texture coordinate', 0)
    lineSrc = socketGetOrNew(grIn.outputs, 'NodeSocketColor', 'line color', 1)
    colorDst = socketGetOrNew(grOut.inputs, 'NodeSocketColor', 'color', 0)

    print("grIn outputs\t%r"% grIn)
    for sock in grIn.outputs:
        print("grIn.outputs['%s'] = %r, %r"%(sock.name, sock.bl_idname, sock))


    y3,x3, node = nodesFor(gr, voronois([0.5, 0.5, 1.0]), -200, 50, uvSrc, lineSrc)

    print(grIn, grOut, node)

    grOut.location = (0,0)

    grIn.location = (x3,0)

    gr.links.new(colorDst, node.outputs[1])

    gr.links.new(node.inputs[0], uvSrc)


def voronois(rgb):
    if False:
        return [
            VoronoiNode((1, 0, 0), (1, 0, 0, 1)),
            VoronoiNode((0, 1, 0), (1, 0, 1, 1)),
            VoronoiNode((0, 0, 0), (0, 1, 0, 1))
        ]


    res=5
    noise = 0.7/res
    colorNoise =0.1

    rval = []
    for u in range(res):
        for v in range(res):
            x = (u+0.5)/res + randPM() * noise
            y = (v+0.5)/res + randPM() * noise
            z=0

            r = rgb[0] + colorNoise*random.random()
            g = rgb[1] + colorNoise*random.random()
            b = rgb[2] + colorNoise*random.random()

            rval.append( VoronoiNode( (x,y,z), (r,g,b,1)))
    return rval



def randPM():
    return (random.random() - 0.5) * 2


def voronoiDiscriminator():
    return bpy.data.node_groups['voronoi discriminator']


mission1()
