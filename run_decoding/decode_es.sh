model_dir='path-to-model'
cluster_dir='path-to-cluster'
out_dir='path-to-out-dir'
coco_dir='path-to-coco-dataset'

coco_model=${model_dir}'best.pkl'
coco_vocab=${model_dir}'coco_vocab.pkl'
split='test'

do_pred_fuse='True'
beam_width='5'
max_len='20'

i=1
max_its=9

# iterate through image cluster files (2, 4 and 9 distractors)
for CLUSTER_FILE in ${cluster_dir}'image_clusters_test_3.pkl' ${cluster_dir}'image_clusters_test_5.pkl' ${cluster_dir}'image_clusters_test_10.pkl'
do

  # iterate through lambda values (0.3, 0.5 and 0.7)
  for LAMBDA in 0.3 0.5 0.7
  do

    # display current iteration
    printf "\n\n#############\n"
    printf "Iteration $i/$max_its"
    printf "\n#############\n\n"

    # execute python file with the arguments defined above
    # and the current values for --coco_cluster and --lambda_
    python decode_captions_es.py  --coco_model $coco_model --coco_vocab $coco_vocab --coco_cluster $CLUSTER_FILE --out_dir $out_dir --image_dir $coco_dir --split $split --lambda_ $LAMBDA --beam_width $beam_width --max_len $max_len

    # advance iteration counter
    i=$(($i+1))

  done

done
