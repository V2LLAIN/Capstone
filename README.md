
```python
cd OCR 
conda env create -f ocr.yml
# pip install -r requirements.txt
python setup.py develop --no_cuda_ext
python restore.py
```


cf) restore.py에서 가장 최초 실행때만 아래 코드 주석해제 후, 실행할 것.
```python
import gdown
gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)
```
