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
<pre><code><b>dataset</b>
├ <b>fracture_detection</b>
│ └ <b>annotations</b>
│   └ xxxxxxxx.npy
└ <b>scaphoid_detection</b>
  ├ <b>annotations</b>
  │ └ xxxxxxxx.npy
  └ <b>images</b>
    └ xxxxxxxx.jpg</code></pre>
