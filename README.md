## 실행 방법

### 1. 환경 준비

Python 3.9 이상 필요

```bash
pip install -r requirements.txt
```

### 2. 결과 재현 (PC)

실측 파라미터로 시뮬레이션을 실행, 결과 표는 `run_output.txt`, 그림은 `figures/`에 저장됨. 난수 seed가 고정되어 있어 항상 같은 결과가 나옴.

```bash
python ekf_edge_battery.py --params params_measured.json > run_output.txt
```

Jetson 실측 전력 그래프(`figures/fig0_jetson_power.png`)를 생성.

```bash
D=data/run_20261025_111741
python parse_tegrastats.py plot --idle $D/idle1.log $D/idle2.log --light $D/light.log --twist $D/twist.log
```

(Optional) 원본 로그에서 실측 파라미터(`params_measured.json`)를 재계산.

```bash
python parse_tegrastats.py build --idle $D/idle1.log $D/idle2.log --light $D/light.log --twist $D/twist.log \
       --counts-light $D/inference_counts_light.csv --counts-twist $D/inference_counts_twist.csv
```

`--params` 없이 실행하면 자리표시자 값으로 동작

### 3. Jetson에서 직접 측정 (Optional)

CUDA를 지원하는 PyTorch, TensorRT Python 바인딩, 그리고 기본 모델을 쓸 경우 torchvision이 필요

1. `inference_logger_template.py`의 `load_student()`와 `INPUT_SHAPE`를 사용할 모델에 맞게 수정. 기본값은 ResNet-18
2. 전원 모드를 고정함. `grep POWER_MODEL /etc/nvpmodel.conf`로 15W 모드의 ID를 확인한 뒤 `sudo nvpmodel -m <ID>`를 실행.
3. 빠른 점검을 수행. 첫 실행 때 TensorRT 엔진이 생성되며, 실제 latency가 출력되는지 확인.
   ```bash
   python3 inference_logger_template.py --mode light --minutes 1 --warmup 10
   python3 inference_logger_template.py --mode twist --minutes 1 --warmup 10
   ```
4. 측정 세션을 실행. 약 50분이 걸리며, 결과는 `run_<날짜>_<시간>/` 폴더에 저장됨.
   ```bash
   tmux new -s measure
   sudo su
   PY=/path/to/your/python bash measure_on_jetson.sh
   ```
   
5. 생성된 폴더를 PC로 옮겨 2번의 `build`와 시뮬레이션을 실행.