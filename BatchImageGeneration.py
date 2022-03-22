import os

# python eval.py 
# --model_file ./zoe-generate/[res+IN]-beta+gamma/[weight=1000]starry-ckpt/fast-style-model.ckpt-done 
# --image_file img/test.jpg 
# --style_strength 0.02

'''
不同strength
'''
model_path = './zoe-generate/[res+IN]-beta+gamma/[weight=220]wave-ckpt/fast-style-model.ckpt-done'
eval_path = 'img/test.jpg'
# strength_list = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
# strength_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
strength_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
# strength_list = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]


for i in range(len(strength_list)):
    code_ = 'python eval.py ' + \
                ' --model_file ' + model_path + \
                ' --image_file ' + eval_path + \
                ' --style_strength ' + str(strength_list[i])
    print("***********************",code_)
    os.system(code_)


'''
不同step
'''
# # setp数量
# generated_num = 1
# # 父级目录
# par_path = 'zoe-generate/[strength2.6-1000]with_parm-IN/'

# # 模型
# model_file = par_path + '[strength2.6-1000]starry-ckpt/'
# model_name = 'fast-style-model.ckpt-' #fast-style-model.ckpt-done / fast-style-model.ckpt-1000

# # 测试图片
# eval_path = 'img/test.jpg'

# # 生成图片
# generated_file = par_path + '[strength2.6-1000]starry-test/'
# generated_name = 'starry-test-' # [model_name]-[test_name]-[step]-[strength]

# for i in range(generated_num):
#     step_str = str(19000+1000*(i+1))
#     if 1:
#     # if i == generated_num-1:
#         model_path = model_file + model_name + 'done'
#     else:
#         model_path = model_file + model_name + step_str
#     for j in range(10):
#         if j == 10:
#             style_strength_str = '1.0'
#         else:
#             style_strength_str = '0.0' + str(1*(j))
#         generated_path = generated_file + generated_name + step_str + '-' + style_strength_str +'.jpg'
#         print("***********************",generated_path)
#         code_ = 'python eval.py --model_file ' + model_path + \
#                     ' --image_file ' + eval_path + \
#                     ' --generated_image_name ' + generated_path + \
#                     ' --style_strength ' + style_strength_str
#         print("***********************",code_)
#         os.system(code_)
