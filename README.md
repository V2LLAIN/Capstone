## cf) SETTING
```python
cd OCR 
conda env create -f ocr.yml
# pip install -r requirements.txt
python setup.py develop --no_cuda_ext
python restore.py
```


### cf-1) restore.py에서 가장 최초 실행때만 아래 코드 주석해제 후, 실행할 것.
```python
import gdown
gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)
```


### cf-2) User가 필수로 적어줘야 하는 항목:
```python
input_path = '/home/work/cargo.jpeg'     # src_img
output_path = '/home/work/cargo_1.jpeg'  # 개선된 img
true_string = "LSCU1077379"  # 정답으로 주어진 11개의 글자    # Cargo ID 정답 Label (Accuracy 계산을 위함)
```

### cf-3) 추가적인 조건
```python
예측된 문자열의 길이가 정답 문자열의 길이와 다르거나, 정확도가 0.7보다 작은 경우 true_string명 출력
이를 통해 수기 작성자가 명확하게 무엇을 먼저 수정해야하는지 확인가능
```
