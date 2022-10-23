k_shot=(0 1 5 10 15 30)
models=("EleutherAI/gpt-neo-125M")
for model in "${models[@]}"
do
  echo "Working on model:"
  echo $model
  for k in "${k_shot[@]}"
  do
    echo "Working on k:"
    echo ludwig/${k}-shot
    python main.py --model_api_name 'hf-causal' --model_args pretrained=${model} --task_name ludwig/${k}-shot  --template_names 'template_1,template_2,template_3,template_4,template_5,template_6' --device cpu
  done
done

