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

/*
cellIdx = cellIdx + patternIdx*cellsPerPattern + layer*patternsPerLayer*cellsPerPattern
xy = cellCenters[ cellIdx ]
patternIdx = patternsForCell[ cellIdx ]
strokeColors[ 3*layer +[0:2] ]
*/
__kernel void voronoi_twisty(float x0, float y0, float dx, float dy,
                             int layerA,
                             global float2 *cellCenters, global int *patternsForCell, global uchar *strokeColors, global uchar *greys,
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
    uchar3 color = (uchar3)(0x80, 0x80, 0x80);
    for (int q=0; q<7; q++) {
        int l2 = (q+layerA)%nLayers;
        int base = cellsPerPattern*(currPattern+patternsPerLayer*l2);
        float2 cell = voronoi_calc(xy, cellCenters+ base, cellsPerPattern);

        int chosenIdx = floor(cell.x);
        int chosenPattern = patternsForCell[chosenIdx + base];

        if (cell.y<0.05) {
            color = (uchar3)( strokeColors[l2*3],
             strokeColors[l2*3+1],
             strokeColors[l2*3+2] );
            break;
        }
        xy = zoomPerLayer*(xy - cellCenters[base+chosenIdx]);
        currPattern = chosenPattern;
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

class VoronoiTwisty:

    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        source = opencl_kernel()

        self.prg = cl.Program(self.ctx, source).build()

        self.k_voronoi_twisty = self.prg.voronoi_twisty
        self.k_voronoi_twisty.set_scalar_arg_dtypes( [ numpy.float32, numpy.float32, numpy.float32,numpy.float32,
                                                     numpy.int32,
                                                     None, None, None, None,
                                                       numpy.int32, numpy.int32, numpy.int32, numpy.float32,
                                                     None])

    def voronoi_twisty(self, x0, y0, dx, dy, layerA, centers, cellPatterns, strokeColors, greys,
                       cellsPerPattern, patternsPerLayer, layerCount, zoomPerLayer, nCols, nRows):

        jj = cellsPerPattern * patternsPerLayer * layerCount
        centers_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, jj*4*2)
        patterns_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, jj*4)
        pixels_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nRows*nCols*3)
        strokeColors_g  = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, layerCount*3)
        greys_g  = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, layerCount*3)

        centers_ = numpy.asarray(centers, dtype=numpy.float32)
        future = cl.enqueue_copy(self.queue, centers_g, centers_)
        future.wait()
        future = cl.enqueue_copy(self.queue, patterns_g, numpy.asarray(cellPatterns, dtype=numpy.int32) )
        future.wait()
        future = cl.enqueue_copy(self.queue, strokeColors_g, numpy.asarray(strokeColors, dtype=numpy.int8) )
        future.wait()
        future = cl.enqueue_copy(self.queue, greys_g, numpy.asarray(greys, dtype=numpy.int8) )
        future.wait()

        future = self.k_voronoi_twisty(self.queue, [nCols, nRows], None,
            x0, y0, dx, dy, layerA,
            centers_g, patterns_g, strokeColors_g, greys_g,
            cellsPerPattern, patternsPerLayer, layerCount, zoomPerLayer, pixels_g)
        future.wait()

        pixels = numpy.zeros([nRows, nCols, 3], dtype=numpy.uint8)
        future = cl.enqueue_copy(self.queue, pixels, pixels_g)
        future.wait()

        return pixels



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
                    rval.append(random.randint(0, patternsPerLayer))

    return rval


def saveResultImage(fname, pixels):
    from PIL import Image
    print ("image shape %r"%(pixels.shape,))
    img = Image.fromarray(pixels, mode='RGB')
    # imsave("/tmp/x.png", pixels)
    img.save(fname)
    print("wrote %s" % fname)

class DiagramParameters:
    def __init__(self, cellsPerPattern, patternsPerLayer, layerCount):
        self.layerCount = layerCount
        self.patternsPerLayer = patternsPerLayer
        self.cellsPerPattern = cellsPerPattern


def calculateGreys(vt, strokeColors, centers, cellPatterns, dp):
    greys = [colorsys.hsv_to_rgb(i / dp.layerCount + 0.13, 0.7, 0.7) for i in range(dp.layerCount)]
    greys = (numpy.asarray(greys, dtype=numpy.float32) * 255.8).astype(numpy.uint8)

    for q in range(3):
        greys2 = []
        for lk in range(dp.layerCount):
            rgb = vt.voronoi_twisty(-4, -4, 8, 8, lk, centers, cellPatterns, strokeColors, greys,
                                    dp.cellsPerPattern, dp.patternsPerLayer, dp.layerCount, 2.8, 1024, 1024)

            colorMean = numpy.mean(rgb, (0, 1))
            greys2.extend(colorMean)
            colorMean = colorMean/255
            hls = colorsys.rgb_to_hsv(colorMean[0], colorMean[1], colorMean[2])

            print("color mean[%d] %r\t%r" % (lk, hls, colorMean))
        greys = greys2

    return greys2


def downsample(pixels, factor):
    (w,h,q) = pixels.shape
    newShape = [floor(h / factor), factor, floor(w / factor), factor, q]
    print(newShape)
    p2 = numpy.reshape(pixels.astype(numpy.float32), newShape, order="C")

    rval = numpy.mean(p2, (1,3))

    return rval.astype(numpy.uint8)


def mission1() :
    vt = VoronoiTwisty()

    rad=3
    diam=1+rad*2
    cellsPerPattern = diam * diam
    patternsPerLayer = 25
    layerCount = 20

    centers = generateCenters(rad, patternsPerLayer * layerCount)
    cellPatterns = pickCellPatterns(rad, layerCount, patternsPerLayer)

    dp = DiagramParameters(cellsPerPattern, patternsPerLayer, layerCount)

    strokeColors = [ colorsys.hsv_to_rgb( i/layerCount, 1, 1) for i in range(layerCount)]
    strokeColors = ( numpy.asarray(strokeColors, dtype=numpy.float32) * 255.8 ).astype(numpy.uint8)
    greys = calculateGreys(vt, strokeColors, centers, cellPatterns, dp)

    print(greys)

    w = 4096

    zoomPerLayer = 2.8
    stepsPerLayer = 30
    for lk in range(20):
        for j in range(stepsPerLayer):
            fr = lk*stepsPerLayer + j

            fname = fileForFrame(fr)

            if os.path.isfile(fname):
                continue

            r1 = 1/ ( zoomPerLayer ** (j/stepsPerLayer) )
            rgb = vt.voronoi_twisty(-r1, -r1, r1*2, r1*2, lk, centers, cellPatterns, strokeColors, greys,
                                cellsPerPattern, patternsPerLayer, layerCount, zoomPerLayer, w, w)

            rgb = downsample(rgb, 4)

            saveResultImage(fname, rgb)


def fileForFrame(fr):
    return "/var/tmp/blender/2019/voronoi_twisty/%04d.png" % fr


#
#
#

mission1()