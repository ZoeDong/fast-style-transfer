## code
python train.py
python eval.py --model_file ./zoe-generate/[res+IN]-beta+gamma/[weight=1000]starry-ckpt/fast-style-model.ckpt-9000 --image_file img/test.jpg --style_strength 1.0

python eval.py --model_file ./models/denoised_starry-2022-03-20_12-53-04/fast-style-model.ckpt-done --image_file img/test.jpg --style_strength 1.0
zoe-generate\[res+IN]-beta+gamma\[weight=1000]starry-ckpt\fast-style-model.ckpt-9000

## tf1.x
- [content loss: 20w,  original_style_loss:10w ]
> python train.py
step: 10,  total Loss 322975.718750, secs/step: 23.532091, content loss: 203105.781250,  original_style_loss: 14176.591797
step: 20,  total Loss 233813.921875, secs/step: 25.993228, content loss: 318184.937500,  original_style_loss: 16428.240234
step: 30,  total Loss 278713.562500, secs/step: 24.572568, content loss: 208595.984375,  original_style_loss: 16810.753906
step: 40,  total Loss 214178.281250, secs/step: 17.132123, content loss: 261488.406250,  original_style_loss: 15029.663086
step: 50,  total Loss 253928.765625, secs/step: 16.989052, content loss: 186565.984375,  original_style_loss: 16034.775391
step: 60,  total Loss 282164.937500, secs/step: 16.996462, content loss: 249217.609375,  original_style_loss: 14591.095703
step: 70,  total Loss 188565.234375, secs/step: 16.995911, content loss: 166804.187500,  original_style_loss: 13725.594727
step: 80,  total Loss 215308.015625, secs/step: 18.415626, content loss: 226040.312500,  original_style_loss: 13684.008789