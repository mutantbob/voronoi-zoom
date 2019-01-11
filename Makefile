CRF=20

D=/var/tmp/blender/2018

P=$D/voronoi-leaf

P2=/var/tmp/blender/2019/voronoi_twisty

today: $(P2).mp4

$P.mp4: PNGs1
	ffmpeg -y -r 30 -i $P/baked/%04d.png  -vcodec libx264 -qp $(CRF) -pix_fmt yuv420p -f mp4 $@

$(P2).mp4: PNGs10
	ffmpeg -y -r 30 -i $(P2)/%04d.png -vcodec libx264 -qp $(CRF) -pix_fmt yuv420p -f mp4 $@

vt.d:
	( echo PNGs10= $(P2)/{0000..0599}.png; echo 'PNGs10: $$(PNGs10)'; echo; echo '$$(PNGs10)' : ; \
	echo "	"PYOPEN_CTL=\'\' python voronoi_twisty.py ) > $@


include blender.d
include vt.d
