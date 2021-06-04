#!/bin/bash
TAG="cuda-pointsource-512sky_2048beam_offsetreflector-pair"
nside=512
beamnside=2048
#beamfwhm_deg=0.6604
beamfwhm_deg=2.0
scan=wholesky 
det=p
# pol can be Q or U (yes, capitalized)
pol=Q
# valid values are ON and OFF (yes, capitalized)
cuda=ON
########################################################################
#                         do not modifidy below                        #
########################################################################
# compile program
cd build || exit 1
rm -r * || exit 1
cmake ../ \
  -DCONVOLVER_DISABLECHI=OFF \
  -DCONVOLVER_OPENMPENABLED=ON \
  -DPOLBEAM_DUMPBEAMS=ON \
  -DCONVOLVER_DEBUG=ON \
  -DMAPPING_DEBUG=ON \
  -DUSE_CUDA=$cuda || exit 1
make -j4 simulate_pointsource_scan.x || exit 1
cd ../ || exit 1
# build beams
python scripts/fields2beams.py \
  data/beams/offset_ref/det_a.txt $beamnside || exit 1
python scripts/fields2beams_cxisco.py \
  data/beams/offset_ref/det_b.txt $beamnside || exit 1
cp data/beams/offset_ref/hpx_tilde_beams_det_a.txt \
  "output/beam_a.txt" || exit 1
cp data/beams/offset_ref/hpx_tilde_beams_det_b.txt \
  "output/beam_b.txt" || exit 1
# execute simulation
echo "running detector "$det" for wholesky simulation "$TAG
program=simulate_pointsource_scan.x
time "./build/"$program \
  -s $pol \
  -p $det \
  -o "output/MAPS_det_"$det"_scan_"$scan".map" \
  -t g \
  -a $beamfwhm_deg \
  -b $beamfwhm_deg \
   > "output/STDOUT_det_"$det"_scan_"$scan".txt" || exit 1
# move dumped beams to output folder
mv dump_detector_a.txt output/
mv dump_detector_b.txt output/
## plot maps
python scripts/plot_maps.py \
  $nside \
  $pol \
  "output/MAPS_det_"$det"_scan_"$scan".map"
# move things to final directory
#mv *.png output/
mkdir $TAG
cp -r output/* $TAG"/"
# none of this ever happened, gentlemen
#rm -r output/*
#exit 0
