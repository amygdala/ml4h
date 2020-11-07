#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil/ml
git checkout pd_atria
git pull

sudo mkdir -p /mnt/disks/segmented-sax-lax-ml4h-v20201102
sudo mount -o norecovery,discard,defaults /dev/sdb /mnt/disks/segmented-sax-lax-ml4h-v20201102/

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP-1))

for i in $(seq $cnt1 $cnt2)
do
    end=$((i+1))
    /home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/parameterize_rv_geom.py $i $end
done

cd /home/pdiachil/ml/notebooks/mri
/snap/bin/gsutil cp *processed* gs://ml4cvd/pdiachil/rightheart_boundary_40k_ml4h_v20201102/
# /snap/bin/gsutil cp *hd5 gs://ml4cvd/pdiachil/rightheart_boundary_images_v20201102/
# /snap/bin/gsutil cp *xmf gs://ml4cvd/pdiachil/rightheart_boundary_images_v20201102/

cd /home/pdiachil/projects/chambers
/snap/bin/gsutil cp poisson* gs://ml4cvd/pdiachil/rightheart_boundary_xdmf_40k_ml4h_v20201102/
/snap/bin/gsutil cp /home/pdiachil/out* gs://ml4cvd/pdiachil/rightheart_boundary_logs_ml4h_40k/

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}