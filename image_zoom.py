import bpy
from math import *


def get_or_create_fcurve(action, data_path, index):
    for fc in action.fcurves:
        if fc.data_path == data_path and (index is None or fc.array_index == index):
            return fc

    return action.fcurves.new(data_path = data_path, index=index)

def mission1(obj):

    if obj.animation_data is None:
        obj.animation_data_create()

    if obj.animation_data.action is None:
        action = bpy.data.actions.new("zoom")
        obj.animation_data.action = action
    else:
        action = obj.animation_data.action

    makeZoomKeyframes(action)


def mission2():
    makeZoomKeyframes(bpy.data.actions['zoom'])


def makeZoomKeyframes(action):
    fcs = [get_or_create_fcurve(action, "scale", idx) for idx in [0, 1]]
    base = 8 / 3
    fr0 = 1
    framesPerCycle = 30
    res = 2
    count = 3 * res + 1
    for fc in fcs:
        nkp = len(fc.keyframe_points)
        if nkp < count:
            fc.keyframe_points.add(count - nkp)
    for j in range(count):
        fr = fr0 + j / res * framesPerCycle

        scale = base ** (j / res)

        fr1 = fr + framesPerCycle / res / 3
        fr9 = fr - framesPerCycle / res / 3

        for ch in [0, 1]:
            kp = fcs[ch].keyframe_points[j]
            kp.co = (fr, scale)

            slope = 1 / 3 / res * log(base) * scale
            kp.handle_right = (fr1, scale + slope)
            kp.handle_left = (fr9, scale - slope)


#
#
#


#mission1(bpy.context.active_object)
mission2()