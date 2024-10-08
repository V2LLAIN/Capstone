import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import gdown

############################ 모델 가중치 다운로드.##########################
# gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)
######################################################################

######################## User가 적어줘야 하는 사항 ########################  
input_path = '/home/work/d.jpeg'
output_path = '/home/work/d_1.jpeg'
true_string = "LSCU1077379"  # 정답으로 주어진 11개의 글자
######################################################################



def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img, save_path):
      model.feed_data(data={'lq': img.unsqueeze(dim=0)})

      if model.opt['val'].get('grids', False):
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      sr_img = tensor2img([visuals['result']])
      imwrite(sr_img, save_path)

opt_path = 'options/test/SIDD/NAFNet-width64.yml'
opt = parse(opt_path, is_train=False)
opt['dist'] = False
NAFNet = create_model(opt)
img_input = imread(input_path)
inp = img2tensor(img_input)
single_image_inference(NAFNet, inp, output_path)
img_output = imread(output_path)

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().cuda()


# 테스트할 이미지 파일
image_file = output_path
res = model.chat(tokenizer, image_file, ocr_type='ocr')
result = ''.join(res.split())[:11]
print(result)

# 예측된 문자열
predicted_string = result

# Accuracy, F1-Score, Precision, Recall 계산을 위해 문자열을 문자 단위로 리스트 변환
true_chars = list(true_string)
predicted_chars = list(predicted_string)

# 정확도, F1 스코어, 정밀도, 재현율 계산
accuracy = accuracy_score(true_chars, predicted_chars)
f1 = f1_score(true_chars, predicted_chars, average='macro')
precision = precision_score(true_chars, predicted_chars, average='macro')
recall = recall_score(true_chars, predicted_chars, average='macro')

# print(f"Accuracy: {accuracy}")
# print(f"F1-Score: {f1}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")

# 추가 조건: 예측된 문자열의 길이가 정답 문자열의 길이와 다르거나, 정확도가 0.7보다 작은 경우 이미지 파일명 출력
if len(predicted_string) != len(true_string) or accuracy < 0.7:
    print(f"Predicted string length mismatch or low accuracy: {true_string}")
