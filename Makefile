CRF=20

D=/var/tmp/blender/2018

P=$D/voronoi-leaf

P2=/var/tmp/blender/2019/voronoi_twisty
P3=/var/tmp/blender/2019/voronoi_twisty2
P4=/var/tmp/blender/2019/voronoi_twisty3

today: $(P4)-26.mp4

$P.mp4: PNGs1
	ffmpeg -y -r 30 -i $P/baked/%04d.png  -vcodec libx264 -qp $(CRF) -pix_fmt yuv420p -f mp4 $@

$(P2).mp4: PNGs10
	ffmpeg -y -r 30 -i $(P2)/%04d.png -vcodec libx264 -qp $(CRF) -pix_fmt yuv420p -f mp4 $@

$(P3).mp4: PNGs11
	ffmpeg -y -r 30 -i $(P3)/%04d.png -vcodec libx264 -qp $(CRF) -pix_fmt yuv420p -f mp4 $@

$(P4).mp4: PNGs12
	ffmpeg -y -r 30 -i $(P4)/%04d.png -vcodec libx264 -qp $(CRF) -pix_fmt yuv420p -f mp4 $@

$(P4)-26.mp4: PNGs12
	ffmpeg -y -r 30 -i $(P4)/%04d.png -vcodec libx264 -qp 26 -pix_fmt yuv420p -f mp4 $@

$(P4)-26.ts: 
	ffmpeg -y  -i $(P4)-26.mp4 -c copy -f mpegts $@

#$(P4)_oct.mp4: $(P4)-26.mp4
#	ffmpeg -y  -i $< -c copy \
#		-i $< -c copy \
#		-i $< -c copy \
#		-i $< -c copy \
#		-i $< -c copy \
#		-i $< -c copy \
#		-i $< -c copy \
#		-i $< -c copy -f mp4 $@

$(P4)_oct.ts: $(P4)-26.ts 
	cat $< $< $< $< $< $< $< $< > $@

$(P4)_oct.mp4: $(P4)_oct.ts
	ffmpeg -y -i $< -c copy -f mp4 $@

vt.d:
	( echo PNGs10= $(P2)/{0000..1199}.png; \
	echo 'PNGs10: $$(PNGs10)'; \
	echo; \
	echo PNGs11= $(P3)/{0000..2399}.png; \
	echo 'PNGs11: $$(PNGs11)'; \
	echo; \
	echo PNGs12= $(P4)/{0000..1799}.png; \
	echo 'PNGs12: $$(PNGs12)'; \
	echo; \
	echo '$$(PNGs10)' : ; \
	echo "	"PYOPEN_CTL=\'\' python voronoi_twisty.py ) > $@


include blender.d
include vt.d
