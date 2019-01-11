import pyopencl as cl
from math import *
import numpy
import random

def opencl_kernel():
    return """

float2 voronoi_calc(float2 xy, global float2 * cellCenters, int nCells)
{
    int returnIdx=0;
    float2 chosenXY = cellCenters[returnIdx];
    float dist = INFINITY;

    for (int j=1; j<nCells; j++) {
        float2 centerDelta = chosenXY - cellCenters[j];
        float2 normCD = centerDelta/length(centerDelta);
        float d1 = dot(chosenXY - xy, normCD);
        float d2 = dot(xy - cellCenters[j] , normCD);
        if (d2 < d1) {
            returnIdx = j;
            dist = fabs(d2-d1)*0.5;
            chosenXY = cellCenters[returnIdx];
        } else if (j==1) {
            dist = fabs(d2-d1)*0.5;
        }
    }

    return (float2)(returnIdx, dist);
}

/*
cellIdx = cellIdx + patternIdx*cellsPerPattern + layer*patternsPerLayer*cellsPerPattern
xy = cellCenters[ cellIdx ]
patternIdx = patternsForCell[ cellIdx ]
*/
__kernel void voronoi_twisty(float x0, float y0, float dx, float dy,
                             int layerA,
                             global float2 *cellCenters, global int *patternsForCell,
                             int cellsPerPattern, int patternsPerLayer,
                             global uchar*rgb) {
    int u = get_global_id(0);
    int v = get_global_id(1);
    int nCols = get_global_size(0);
    int nRows = get_global_size(1);
    int idx = u+v*nCols;

    float2 xy = (float2) (
    x0 + dx * u / nCols,
    y0 + dy * v / nRows );

    int base = cellsPerPattern*(0+patternsPerLayer*layerA);
    float2 cell = voronoi_calc(xy, cellCenters+ base, cellsPerPattern);

    int chosenIdx = floor(cell.x);
    int chosenPattern = patternsForCell[chosenIdx + base];

    if (cell.y<0.1) {
        rgb[idx*3] = 255;
        rgb[idx*3+1] = 0;
        rgb[idx*3+2] = 0;
    } else {
        float2 xy2 = 2*(xy - cellCenters[base+chosenIdx]);
        base = cellsPerPattern*(chosenPattern + patternsPerLayer*(1+layerA));
        float2 cell2 = voronoi_calc(xy2, cellCenters + base, cellsPerPattern);

        if (cell2.y<0.1) {
            rgb[idx*3] = 0;
            rgb[idx*3+1] = 0;
            rgb[idx*3+2] = 255;
        } else {
            rgb[idx*3] = 0;
            rgb[idx*3+1] =  5*(uchar)chosenIdx;
            rgb[idx*3+2] = 0;
        }
    }
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
                                                     None, None, numpy.int32, numpy.int32,
                                                     None])

    def voronoi_twisty(self, x0, y0, dx, dy, layerA, centers, cellPatterns,
                       cellsPerPattern, patternsPerLayer, layerCount, nCols, nRows):

        jj = cellsPerPattern * patternsPerLayer * layerCount
        centers_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, jj*4*2)
        patterns_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, jj*4)
        pixels_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nRows*nCols*3)

        centers_ = numpy.asarray(centers, dtype=numpy.float32)
        future = cl.enqueue_copy(self.queue, centers_g, centers_)
        future.wait()
        future = cl.enqueue_copy(self.queue, patterns_g, numpy.asarray(cellPatterns, dtype=numpy.int32) )

        future = self.k_voronoi_twisty(self.queue, [nCols, nRows], None,
                             x0, y0, dx, dy, layerA,
                                     centers_g, patterns_g, cellsPerPattern, patternsPerLayer, pixels_g)
        future.wait()

        pixels = numpy.zeros([nRows, nCols, 3], dtype=numpy.int8)
        future = cl.enqueue_copy(self.queue, pixels, pixels_g)
        future.wait()

        return pixels



def generateCenters(diam, nPatterns):

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


#
#
#

vt = VoronoiTwisty()

rad=3
diam=1+rad*2
cellsPerPattern = diam * diam
patternsPerLayer = 25
layerCount = 20

centers = generateCenters(rad, patternsPerLayer * layerCount)
cellPatterns = pickCellPatterns(rad, layerCount, patternsPerLayer)

rgb = vt.voronoi_twisty(-4,-4, 8, 8, 0, centers, cellPatterns, cellsPerPattern, patternsPerLayer, layerCount, 1024, 1024)

saveResultImage("/tmp/x.png", rgb)