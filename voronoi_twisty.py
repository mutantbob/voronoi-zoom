import pyopencl as cl
from math import *
import numpy
import random
import colorsys
import os

def opencl_kernel():
    return """

float2 voronoi_calc(float2 xy, global float2 * cellCenters, int nCells)
{
    int returnIdx=0;
    float2 chosenXY = cellCenters[returnIdx];
    float dist = length(xy-chosenXY);

    for (int j=1; j<nCells; j++) {
        float2 c2 = cellCenters[j];
        float d9 = length(c2 - xy);

        if (d9 < dist) {
            chosenXY = c2;
            dist = d9;
            returnIdx=j;
        }
    }

    float stroke = INFINITY;

    for (int j=0; j<nCells; j++) {
        if (j==returnIdx)
            continue;

        float2 c2 = cellCenters[j];

        float2 centerDelta = chosenXY - c2;
        float2 normCD = centerDelta/length(centerDelta);
        float d1 = dot(chosenXY - xy, normCD);
        float d2 = dot(xy - c2 , normCD);

        float t2_ = (d2-d1)*0.5;
        float t2 = fabs(t2_);

        if (t2<stroke) {
            stroke = t2;
        }
    }

    return (float2)(returnIdx, stroke);
}

float2 rotated(float2 xy, float sinTheta, float cosTheta)
{
    return (float2)(
    xy.x*cosTheta - xy.y*sinTheta,
    xy.x*sinTheta + xy.y*cosTheta
    );
}

#define SIN_COS(ci) (sinecosines + 2*(ci + cellsPerPattern * (currPattern + patternsPerLayer*l2)) )
/*
cellIdx = cellIdx + patternIdx*cellsPerPattern + layer*patternsPerLayer*cellsPerPattern
xy = cellCenters[ cellIdx ]
patternIdx = patternsForCell[ cellIdx ]
(sin(theta), cos(theta)) = rotations[ cellIdx*2 + [0:1] ]
strokeColors[ 3*layer +[0:2] ]
*/
__kernel void voronoi_twisty(float x0, float y0, float dx, float dy,
                             int layerA,
                             global float2 *cellCenters, global int *patternsForCell, global float *sinecosines,
                             global uchar *strokeColors, global uchar *greys,
                             int cellsPerPattern, int patternsPerLayer, int nLayers, float zoomPerLayer,
                             global uchar*rgb) {
    int u = get_global_id(0);
    int v = get_global_id(1);
    int nCols = get_global_size(0);
    int nRows = get_global_size(1);
    int idx = u+v*nCols;

    float2 xy = (float2) (
    x0 + dx * u / nCols,
    y0 + dy * v / nRows );

    int currPattern = 0;
    int chosenIdx = cellsPerPattern/2; // middle cell
    uchar3 color = (uchar3)(0x80, 0x80, 0x80);
    for (int q=0; q<7; q++) {
        int l2 = (q+layerA)%nLayers;
        int base = cellsPerPattern*(currPattern+patternsPerLayer*l2);
        global float* sincos=SIN_COS(chosenIdx);
        float2 xyr = rotated(xy, sincos[0], sincos[1] );
        float2 cell = voronoi_calc(xyr, cellCenters+ base, cellsPerPattern);

        chosenIdx = floor(cell.x);
        currPattern = patternsForCell[chosenIdx + base];

        if (cell.y<0.05) {
            color = (uchar3)( strokeColors[l2*3],
             strokeColors[l2*3+1],
             strokeColors[l2*3+2] );
            break;
        }
        xy = zoomPerLayer*( xyr-cellCenters[base+chosenIdx] );
        xy = rotated( xy, -sincos[0], sincos[1]);

        color = (uchar3)(
            greys[l2*3],
            greys[l2*3+1],
            greys[l2*3+2] );
        //color = (uchar3)(0, 5*(uchar)chosenIdx, 0);
    }

    rgb[idx*3] = color.x;
    rgb[idx*3+1] = color.y;
    rgb[idx*3+2] = color.z;
}
"""

def allocCopyWait(ctx, queue, nArray, flags=cl.mem_flags.READ_WRITE):
    rval = cl.Buffer(ctx, flags, nArray.nbytes)
    future = cl.enqueue_copy(queue, rval, nArray)
    future.wait()
    return rval

class VoronoiTwisty:

    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        source = opencl_kernel()

        self.prg = cl.Program(self.ctx, source).build()

        self.k_voronoi_twisty = self.prg.voronoi_twisty
        self.k_voronoi_twisty.set_scalar_arg_dtypes( [ numpy.float32, numpy.float32, numpy.float32,numpy.float32,
                                                     numpy.int32,
                                                     None, None, None, None, None,
                                                       numpy.int32, numpy.int32, numpy.int32, numpy.float32,
                                                     None])

    def voronoi_twisty(self, x0, y0, dx, dy, layerA, centers, cellPatterns, rotations, strokeColors, greys,
                       cellsPerPattern, patternsPerLayer, layerCount, zoomPerLayer, nCols, nRows):

        sinecosines = [ [sin(theta), cos(theta)] for theta in rotations]
        #print(" sc shape %r"%(numpy.asarray(sinecosines, dtype=numpy.float32).shape, ))
        #print(numpy.asarray(sinecosines, dtype=numpy.float32))
        centers_g = allocCopyWait(self.ctx, self.queue, numpy.asarray(centers, dtype=numpy.float32))
        patterns_g = allocCopyWait(self.ctx, self.queue, numpy.asarray(cellPatterns, dtype=numpy.int32))
        sinecosines_g = allocCopyWait(self.ctx, self.queue, numpy.array(sinecosines, dtype=numpy.float32))
        strokeColors_g = allocCopyWait(self.ctx, self.queue, numpy.asarray(strokeColors, dtype=numpy.int8))
        greys_g = allocCopyWait(self.ctx, self.queue, numpy.asarray(greys, dtype=numpy.int8))
        pixels_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nRows*nCols*3)

        future = self.k_voronoi_twisty(self.queue, [nCols, nRows], None,
            x0, y0, dx, dy, layerA,
            centers_g, patterns_g, sinecosines_g, strokeColors_g, greys_g,
            cellsPerPattern, patternsPerLayer, layerCount, zoomPerLayer, pixels_g)
        future.wait()

        pixels = numpy.zeros([nRows, nCols, 3], dtype=numpy.uint8)
        future = cl.enqueue_copy(self.queue, pixels, pixels_g)
        future.wait()

        centers_g.release()
        patterns_g.release()
        sinecosines_g.release()
        strokeColors_g.release()
        greys_g.release()
        pixels_g.release()

        return pixels


class RotationsAnimator:

    def __init__(self, diagramParams, framesPerLayer):
        self.framesPerLayer = framesPerLayer
        self.dp = diagramParams
        n = self.dp.cellsPerPattern*self.dp.patternsPerLayer*self.dp.layerCount
        self.velocities = [ self.randomVelocity() for i in range(n)]
        self.baseline = [ random.random()*2*pi for i in range(n)]

    def randomVelocity(self):
        a = random.random() - 0.5
        if (a<0):
            a = a-0.5
        else:
            a = a+0.5
        return 0.1 / 30 * 2 * (a)

    def rotationForSlot(self, slot, frame):

        currLayer = floor(frame / self.framesPerLayer)
        slotLayer = floor(slot/(self.dp.patternsPerLayer
                                 *self.dp.cellsPerPattern
                                #*self.dp.layerCount
                                ))

        layerCount = self.dp.layerCount
        oppositeLayer = (slotLayer + floor(layerCount / 2)) % layerCount
        frame0 = oppositeLayer * self.framesPerLayer
        frame9 = (slotLayer+1)*self.framesPerLayer

        if False and (slotLayer>10):
            print ("frame %d\t slot end %d\t"%(frame, frame9))
        delta = self.wrapFrame(frame, frame9)

        return delta * self.velocities[slot] + self.baseline[slot]

    def wrapFrame(self, frame, frame0):
        if frame > frame0:
            delta = frame0 - frame + self.framesPerLayer * self.dp.layerCount
        else:
            delta = frame0 - frame
        return delta

    def rotationsForFrame(self, fr):
        return [ self.rotationForSlot(i, fr) for i in range(len(self.velocities))]
        #return [ fr* v for v in self.velocities ]


def generateCenters(rad, nPatterns):

    diam = 1+2*rad
    rval = []

    scatter = 0.6

    for p in range(nPatterns):
        for u in range(-rad, rad+1):
            for v in range(-rad, rad + 1):
                if u==0 and v==0:
                    (x,y) = (0,0)
                else:
                    x = u + (random.random()-0.5) *2 *scatter
                    y = v + (random.random() - 0.5) *2 *scatter

                if 1 >= abs(v):
                    if u==-1:
                        x = min(-1, x)
                    elif u==1:
                        x = max(1, x)
                if 1>=abs(u):
                    if -1 == v:
                        y = min(-1, y)
                    elif 1==v:
                        y = max(1, y)

                rval.extend( [x,y])

    return rval


def pickCellPatterns(rad, layerCount, patternsPerLayer):
    diam = 1+2*rad
    center = floor(diam*diam/2)
    rval = []
    for l in range(layerCount):
        for p in range(patternsPerLayer):
            for i in range(diam*diam):
                if i==center:
                    rval.append(0)
                else:
                    rval.append(random.randint(0, patternsPerLayer-1))

    return rval


def saveResultImage(fname, pixels):
    from PIL import Image
    #print ("image shape %r"%(pixels.shape,))
    img = Image.fromarray(pixels, mode='RGB')
    # imsave("/tmp/x.png", pixels)

    tmp = fname+".new"
    img.save(tmp, format="PNG")

    import  os
    import contextlib
    with contextlib.suppress(FileNotFoundError):
        os.remove(fname)
    os.rename(tmp, fname)

    print("wrote %s" % fname)

class DiagramParameters:
    def __init__(self, cellsPerPattern, patternsPerLayer, layerCount):
        self.layerCount = layerCount
        self.patternsPerLayer = patternsPerLayer
        self.cellsPerPattern = cellsPerPattern


def calculateGreys(vt, strokeColors, centers, cellPatterns, rotations, dp, zoomPerLayer):
    greys = [colorsys.hsv_to_rgb(i / dp.layerCount + 0.13, 0.7, 0.7) for i in range(dp.layerCount)]
    greys = (numpy.asarray(greys, dtype=numpy.float32) * 255.8).astype(numpy.uint8)

    for q in range(3):
        greys2 = []
        for lk in range(dp.layerCount):
            rgb = vt.voronoi_twisty(-2, -2, 4, 4, lk, centers, cellPatterns, rotations, strokeColors, greys,
                                    dp.cellsPerPattern, dp.patternsPerLayer, dp.layerCount, zoomPerLayer, 1024, 1024)

            colorMean = numpy.mean(rgb, (0, 1))
            greys2.extend(colorMean)
            colorMean = colorMean/255
            hls = colorsys.rgb_to_hsv(colorMean[0], colorMean[1], colorMean[2])

            if (lk==5):
                print("color mean[%d] %r\t%r" % (lk, hls, colorMean))
        greys = greys2

    return greys2


def downsample(pixels, factor):
    (h,w,q) = pixels.shape
    newShape = [floor(h / factor), factor, floor(w / factor), factor, q]
    #print(newShape)
    p2 = numpy.reshape(pixels.astype(numpy.float32), newShape, order="C")

    rval = numpy.mean(p2, (1,3))

    return rval.astype(numpy.uint8)


def mission1() :
    vt = VoronoiTwisty()

    rad=3
    cellsPerPattern = (1 + rad * 2) ** 2

    dp = DiagramParameters(cellsPerPattern, 25, 20)

    renderAnimation(vt, dp, 60, 2.8, fileForFrame)


def mission3() :
    vt = VoronoiTwisty()

    rad=7
    cellsPerPattern = (1 + rad * 2) ** 2

    dp = DiagramParameters(cellsPerPattern, 25, 20)

    renderAnimation(vt, dp, 120, 4.5,
                    lambda fr: "/var/tmp/blender/2019/voronoi_twisty2/%04d.png"%fr)


def mission4() :
    vt = VoronoiTwisty()

    rad=6
    cellsPerPattern = (1 + rad * 2) ** 2

    dp = DiagramParameters(cellsPerPattern, 25, 20)

    renderAnimation(vt, dp, 90, 3.5,
                    lambda fr: "/var/tmp/blender/2019/voronoi_twisty3/%04d.png"%fr)


def renderAnimation(vt, dp, stepsPerLayer, zoomPerLayer, filenameGenerator):

    rad = floor( sqrt(dp.cellsPerPattern) / 2)

    centers = generateCenters(rad, dp.patternsPerLayer * dp.layerCount)
    cellPatterns = pickCellPatterns(rad, dp.layerCount, dp.patternsPerLayer)
    rotAnim = RotationsAnimator(dp, stepsPerLayer)

    strokeColors = [colorsys.hsv_to_rgb(i / dp.layerCount, 1, 1) for i in range(dp.layerCount)]
    strokeColors = (numpy.asarray(strokeColors, dtype=numpy.float32) * 255.8).astype(numpy.uint8)
    greys = calculateGreys(vt, strokeColors, centers, cellPatterns, rotAnim.rotationsForFrame(0), dp, zoomPerLayer)
    print(greys)

    oversample = 4
    w = 1920 * oversample
    h = 1080 * oversample

    for lk in range(dp.layerCount):
        for j in range(0, stepsPerLayer, 1):
            fr = lk * stepsPerLayer + j

            fname = filenameGenerator(fr)

            if os.path.isfile(fname):
                continue

            rotations = rotAnim.rotationsForFrame(fr)

            dx = 1 / (zoomPerLayer ** (j / stepsPerLayer))
            dy = h * dx / w
            rgb = vt.voronoi_twisty(-dx, -dy, dx * 2, dy * 2, lk, centers, cellPatterns, rotations, strokeColors, greys,
                                    dp.cellsPerPattern, dp.patternsPerLayer, dp.layerCount, zoomPerLayer, w, h)

            rgb = downsample(rgb, oversample)

            saveResultImage(fname, rgb)


def fileForFrame(fr):
    return "/var/tmp/blender/2019/voronoi_twisty/%04d.png" % fr


def mission2():
    """ test for my rotation calculation """
    dp = DiagramParameters(49, 25, 20)
    rotAnim = RotationsAnimator(dp, 30)

    old =0
    for fr in range(0, 600, 5):
        rot = rotAnim.rotationForSlot(41, fr)
        delta  =rot-old
        print("%d\t%.5f\t%r" % (fr, rot, delta))

        old=rot

#
#
#

random.seed(4262)
#mission2()

#mission1()
#mission3()
mission4()