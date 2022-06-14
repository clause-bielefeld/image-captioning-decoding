model_dir='path-to-model'
cluster_dir='path-to-cluster'
out_dir='path-to-out-dir'
coco_dir='path-to-coco-dataset'

coco_model=${model_dir}'best.pkl'
coco_vocab=${model_dir}'coco_vocab.pkl'
coco_cluster=${cluster_dir}'image_clusters_test_3.pkl'
split='test'

beam_width='5'
max_len='20'

i=1
max_its=78

# iterate through temperature values (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
for TEMPERATURE in 0.7 0.8 0.9 1.0 1.1 1.2
do

  # iterate through top-p values (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  for TOPP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do

    # display current iteration
    printf "\n\n#############\n"
    printf "Iteration $i/$max_its"
    printf "\n#############\n\n"

    # execute python file with the arguments defined above
    python decode_captions.py  --coco_model $coco_model --coco_vocab $coco_vocab --coco_cluster $coco_cluster --out_dir $out_dir --image_dir $coco_dir --split $split --beam_width $beam_width --max_len $max_len --do_nucleus --top_p $TOPP --temperature $TEMPERATURE

    # advance iteration counter
    i=$(($i+1))

  done

  # iterate through top_k values (5, 10, 25, 50)
  for TOPK in 5 10 25 50
  do

    # display current iteration
    printf "\n\n#############\n"
    printf "Iteration $i/$max_its"
    printf "\n#############\n\n"

    # execute python file with the arguments defined above
    python decode_captions.py  --coco_model $coco_model --coco_vocab $coco_vocab --coco_cluster $coco_cluster --out_dir $out_dir --image_dir $coco_dir --split $split --beam_width $beam_width --max_len $max_len --do_topk --top_k $TOPK --temperature $TEMPERATURE

    # advance iteration counter
    i=$(($i+1))

  done

done
