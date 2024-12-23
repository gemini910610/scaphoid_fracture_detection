# Scaphoid Fracture Detection
## 原始資料夾架構
<pre><code><b>data</b>
├ <b>fracture_detection</b>
│ └ <b>annotations</b>
│   └ xxxxxxxx.json
└ <b>scaphoid_detection</b>
  ├ <b>annotations</b>
  │ └ xxxxxxxx.json
  └ <b>images</b>
    └ xxxxxxxx.jpg</code></pre>
## 整理資料
```python=
python make_dataset.py
```
### 資料夾架構
<pre><code><b>dataset</b>
├ <b>train</b>
│ └ xxxxxxxx.json
└ <b>val</b>
  └ xxxxxxxx.json</code></pre>
### 檔案內容
```javascript=
{
  "image": "xxxxxxxx.jpg",
  "bboxes": {
    "scaphoid": [x1, y1, x2, y2],
    "fracture": [
      [x1, y1],
      [x2, y2],
      [x3, y3],
      [x4, y4]
    ]
  }
}
```
## 舟骨偵測
## 骨折偵測
