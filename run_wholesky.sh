#!/bin/bash
TAG="256-sky_512-beam_offsetreflector-pair-cuda"
nside=128
beamnside=2048
beamfwhm_deg=0.6604
scan=wholesky 
det=p
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
make -j4 simulate_wholesky_scan.x || exit 1
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
# build sky
python scripts/create_cmb.py $nside || exit 1
mv input_dl.png output/ || exit 1
mv input_maps.png output/ || exit 1
mv input_ps.png output/ || exit 1
mv cmb.txt output/ || exit 1
cp data/cls/cl_data.csv output/ || exit 1
mv cl_data.csv output/ || exit 1
# execute simulation
echo "running detector "$det" for wholesky simulation "$TAG
program=simulate_wholesky_scan.x
time "./build/"$program \
  -m "output"/cmb.txt \
  -p $det \
  -o "output/MAPS_det_"$det"_scan_"$scan".map" \
  -t f \
  -a output/beam_a.txt \
  -b output/beam_b.txt \
   > "output/STDOUT_det_"$det"_scan_"$scan".txt" || exit 1
# move dumped beams to output folder
mv dump_detector_a.txt output/
mv dump_detector_b.txt output/
## compute power spectra
python scripts/compute_powerspectrum.py \
  output/cmb.txt \
  "output/MAPS_det_"$det"_scan_"$scan".map" \
  "output/"dump_detector_a.txt $beamnside $beamfwhm_deg
# move resulting power spectra to output folder
mv ps.csv output/
# plot power spectra
python scripts/plot_powerspectrum.py output/ps.csv output/cl_data.csv
# move things to final directory
mv ps.png output/
mkdir $TAG
cp -r output/* $TAG"/"
# none of this ever happened, gentlemen
rm -r output/*
exit 0
