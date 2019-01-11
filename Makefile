CRF=20

D=/var/tmp/blender/2018

P=$D/voronoi-leaf

$P.mp4: PNGs1
	ffmpeg -y -r 30 -i $P/baked/%04d.png  -vcodec libx264 -qp $(CRF) -pix_fmt yuv420p -f mp4 $@

include blender.d
