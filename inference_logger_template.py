"""
inference_logger_template.py -- run the inference workload at a fixed rate while tegrastats logs
power, and record how many inferences actually FINISHED in each minute.

  --mode light : KD student as a TensorRT FP16 engine                    (normal operation)
  --mode twist : the same student in plain PyTorch FP32                  (silent fallback)
                 --twist-kind xai : forward + gradient explanation per frame (XAI audit mode)

Only load_student() (and INPUT_SHAPE) are specific to you. By default it builds torchvision
ResNet-18 with random weights: inference cost does not depend on the weight values, so this is a
valid stand-in when your own student is not at hand (state it in the report).

The TensorRT engine is built once from an ONNX export with trtexec (takes a few minutes the first
time) and cached next to this file as student_fp16.engine. Delete student.onnx and
student_fp16.engine after changing the model or INPUT_SHAPE.

Quick check before the full session (second terminal: sudo tegrastats --interval 1000):
  python3 inference_logger_template.py --mode light --minutes 1 --warmup 10
  python3 inference_logger_template.py --mode twist --minutes 1 --warmup 10
The printed latency must be a real inference time and VDD_IN must rise above idle.

Each inference waits for the GPU to finish (stream / device synchronize), so the count is the
number of finished inferences, not launches. --dry replaces the model by sleeps (TESTS ONLY).
"""
import os
import csv
import time
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_SHAPE = (1, 3, 224, 224)
ONNX_PATH = os.path.join(HERE, "student.onnx")
ENGINE_PATH = os.path.join(HERE, "student_fp16.engine")
STATE = {}


def load_student():
    """Return your KD student as an eval-mode torch.nn.Module in FP32 on the CPU. To use your own:
        model = MyStudent(...)
        model.load_state_dict(torch.load("student.pth", map_location="cpu"))
        return model.eval()
    """
    try:
        import torchvision
    except ImportError:
        raise SystemExit("torchvision is not installed: put your own student model in load_student()")
    return torchvision.models.resnet18(weights=None).eval()


# ----------------------------------------------------------------------------- light: TensorRT FP16
def build_engine():
    import torch
    if not os.path.exists(ONNX_PATH):
        print("  exporting ONNX ...", flush=True)
        torch.onnx.export(load_student(), torch.randn(*INPUT_SHAPE), ONNX_PATH, opset_version=17,
                          input_names=["input"], output_names=["output"])
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        raise SystemExit("trtexec not found (expected /usr/src/tensorrt/bin/trtexec)")
    print("  building the TensorRT FP16 engine (one time, a few minutes) ...", flush=True)
    r = subprocess.run([trtexec, f"--onnx={ONNX_PATH}", "--fp16", f"--saveEngine={ENGINE_PATH}"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0 or not os.path.exists(ENGINE_PATH):
        print(r.stdout[-3000:])
        raise SystemExit("trtexec failed (see the output above)")


def setup_light():
    import torch
    import tensorrt as trt
    if not os.path.exists(ENGINE_PATH):
        build_engine()
    with open(ENGINE_PATH, "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    tmap = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32, trt.DataType.INT8: torch.int8, trt.DataType.BOOL: torch.bool}
    if hasattr(trt.DataType, "INT64"):
        tmap[trt.DataType.INT64] = torch.int64
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    is_in = {n: engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in names}
    for n in names:                                   # dynamic input shapes -> fix them first
        if is_in[n] and -1 in tuple(engine.get_tensor_shape(n)):
            ctx.set_input_shape(n, INPUT_SHAPE)
    bufs = {}
    for n in names:
        shape = tuple(ctx.get_tensor_shape(n))
        bufs[n] = torch.empty(shape, dtype=tmap[engine.get_tensor_dtype(n)], device="cuda")
        if is_in[n]:
            bufs[n].copy_(torch.randn(shape))         # one fixed input, no camera
        ctx.set_tensor_address(n, bufs[n].data_ptr())
    torch.cuda.synchronize()                          # input copy finished before the first run
    STATE.update(engine=engine, ctx=ctx, bufs=bufs, stream=torch.cuda.Stream())
    print(f"  TensorRT engine ready on {torch.cuda.get_device_name(0)}: "
          + ", ".join(f"{n}{tuple(b.shape)} {str(b.dtype).replace('torch.', '')}" for n, b in bufs.items()),
          flush=True)


def infer_light():
    s = STATE["stream"]
    STATE["ctx"].execute_async_v3(s.cuda_stream)
    s.synchronize()


# ----------------------------------------------------------------------------- twist: PyTorch FP32 / XAI
def setup_twist(kind):
    import torch
    STATE.update(model=load_student().float().cuda(), x=torch.randn(*INPUT_SHAPE, device="cuda"), kind=kind)
    print(f"  PyTorch FP32 model ready on {torch.cuda.get_device_name(0)} ({kind})", flush=True)


def infer_twist():
    import torch
    m, x = STATE["model"], STATE["x"]
    if STATE["kind"] == "xai":                        # forward + gradient of the top score (saliency)
        m.zero_grad(set_to_none=True)
        xi = x.detach().requires_grad_(True)
        out = m(xi)
        y = out if torch.is_tensor(out) else out[0]
        y.flatten().max().backward()
    else:                                             # plain FP32 forward
        with torch.inference_mode():
            m(x)
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------- loop
def paced(fn, period, until, on_done=None):
    """Call fn at most once per period until time `until`; never catch up when late."""
    next_t = time.time()
    while time.time() < until:
        t0 = time.perf_counter()
        fn()
        if on_done:
            on_done((time.perf_counter() - t0) * 1000.0)
        next_t += period
        s = next_t - time.time()
        if s > 0:
            time.sleep(s)
        else:
            next_t = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["light", "twist"], required=True)
    ap.add_argument("--twist-kind", choices=["fp32", "xai"], default="fp32")
    ap.add_argument("--fps", type=float, default=30.0, help="target rate")
    ap.add_argument("--minutes", type=int, default=11, help="counted minutes")
    ap.add_argument("--warmup", type=float, default=60.0, help="uncounted warm-up [s]")
    ap.add_argument("--ready-file", help="created when the warm-up is over (used by measure_on_jetson.sh)")
    ap.add_argument("--dry", action="store_true", help="TESTS ONLY: sleeps instead of a model")
    a = ap.parse_args()
    period = 1.0 / a.fps

    print(f"[{a.mode}] loading ...", flush=True)
    if a.dry:
        print("  DRY RUN: no model, no GPU work -- never use these numbers", flush=True)
        fn = (lambda: time.sleep(0.004)) if a.mode == "light" else (lambda: time.sleep(0.012))
    elif a.mode == "light":
        setup_light()
        fn = infer_light
    else:
        setup_twist(a.twist_kind)
        fn = infer_twist

    lat_w = []
    print(f"[{a.mode}] warm-up {a.warmup:.0f} s (not counted) ...", flush=True)
    paced(fn, period, time.time() + a.warmup, on_done=lat_w.append)
    if lat_w:
        tail = lat_w[-min(len(lat_w), 50):]
        print(f"[{a.mode}] warm-up done: {len(lat_w)} inferences, recent latency {sum(tail)/len(tail):.2f} ms",
              flush=True)
    if a.ready_file:
        with open(a.ready_file, "w") as rf:
            rf.write(f"{time.time():.3f}\n")

    out = f"inference_counts_{a.mode}.csv"
    n = None
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minute", "count", "mean_latency_ms"])
        for minute in range(a.minutes):
            lat = []
            paced(fn, period, time.time() + 60.0, on_done=lat.append)
            n = len(lat)
            w.writerow([minute, n, round(sum(lat) / max(n, 1), 2)])
            f.flush()
            print(f"  minute {minute}: {n} inferences ({n/60:.1f} fps), mean latency {sum(lat)/max(n,1):.2f} ms",
                  flush=True)
    print("wrote", out)
    if n is not None and n < 0.95 * a.fps * 60:
        print(f"NOTE: the {a.mode} path did not hold {a.fps:.0f} fps; fine, the parser uses the achieved rate.")


if __name__ == "__main__":
    main()
