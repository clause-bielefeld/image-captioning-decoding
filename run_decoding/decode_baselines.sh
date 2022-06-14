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

# execute python file with the arguments defined above
python decode_captions.py  --coco_model $coco_model --coco_vocab $coco_vocab --coco_cluster $coco_cluster --out_dir $out_dir --image_dir $coco_dir --split $split --do_beam --do_greedy --beam_width $beam_width --max_len $max_len
