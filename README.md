# Scaphoid Fracture Detection
## 原始資料架構
<pre><code><b>data</b>
├ <b>fracture_detection</b>
│ └ <b>annotations</b>
│   └ xxxxxxxx.json
└ <b>scaphoid_detection</b>
  ├ <b>annotations</b>
  │ └ xxxxxxxx.json
  └ <b>images</b>
    └ xxxxxxxx.jpg</code></pre>
## 製作資料集
```python=
python make_dataset.py
```
* 重新命名檔案
* 將骨折偵測的座標換算為全局座標
* 將骨折偵測的座標正歸化到[0, 1]
<pre><code><b>dataset</b>
├ <b>fracture_detection</b>
│ └ <b>annotations</b>
│   └ xxxxxxxx.pth
└ <b>scaphoid_detection</b>
  ├ <b>annotations</b>
  │ └ xxxxxxxx.pth
  └ <b>images</b>
    └ xxxxxxxx.jpg</code></pre>
## 資料預處理
### 舟骨偵測
1. CLAHE
2. 隨機水平翻轉
3. 隨機對比度調整
4. 調整影像大小為(1400, 1200)
