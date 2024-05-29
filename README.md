# DT-Mem
This is the official implementation of paper "Think Before You Act: Decision Transformers with Working Memory"
## Training
To train the model, run this command:
```
python train.py --create_hnet=$create_hnet --max_epochs=1000 --eval_freq 10 --n_embd=512 --n_layer=4 --use_wandb=0\
  --n_head=8 --device='cuda' --n_gpus --num_workers=10 --data_steps $data_steps --training_samples=$samples --use_gw=1\
  --load_path=$model_path --num_datasets 1 --folder_prefix='./' --batch_size=8 --eval=0 --train=1 --apply_lora=$lora\
  --train_game_list $game_name\
  --eval_game_list $game_name\
```

## Fine-tuning
To fine-tune the model, simply change the above code ```--apply_lora=1```

## Contributing
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

