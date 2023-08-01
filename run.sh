# nohup bash -u run.sh > generate_adversarial_examples_for_multishape_and_different_models_23xxxx.out 2>&1 &
TARGET_NETS="resnet50 vgg16 densenet121 resnext50 wideresnet squeezenet"
SHAPE_TYPE="triangle rectangle pentagon hexagon line"

for target_net in $TARGET_NETS; do
  for shape_type in $SHAPE_TYPE; do
    python3 pos_reflect_attack.py \
      --model_name $target_net \
      --shape_type $shape_type
done
done
