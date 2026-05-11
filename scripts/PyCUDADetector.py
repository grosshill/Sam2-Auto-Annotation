import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import time
def func_timer(func, tag='Detector'):
    def wrapper(*args, **kwargs):
        self = args[0]
        debug = getattr(self, 'debug', False)
        if debug: start = time.perf_counter()
        result = func(*args, **kwargs)
        if debug: print(f'{tag}: function {func.__name__} takes {(time.perf_counter() - start) * 1000:.2f}ms')
        return result
    return wrapper

class CUDADetector:
    """
        CUDA detector for TensorRT engine.

        Current expected raw output (without NMS): (1, 5, 8400)
        where 5 channels are [cx, cy, w, h, conf], and class count is fixed to 1.
    """
    def __init__(
        self,
        engine_path,
        debug=False,
        enable_nms=False,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        pre_nms_topk=1000,
    ):
        self.debug = debug
        self.enable_nms = enable_nms
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.pre_nms_topk = int(pre_nms_topk)
        self.TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
        if debug:
            self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_idx = self.engine.get_tensor_name(0)
        self.output_idx = self.engine.get_tensor_name(1)

        self.input_shape = self.engine.get_tensor_shape(self.input_idx)
        self.output_shape = self.engine.get_tensor_shape(self.output_idx)

        if debug:
            print(f"Input shape: {self.input_shape}")
            print(f"Output shape: {self.output_shape}")

        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize))
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize))

        self.stream = cuda.Stream()

    def detect(self, data: np.ndarray) -> np.ndarray:
        # start = time.time()
        # if self.debug:
            # print(f"Input data type: {data.dtype}")
            # print(f"Is data contiguous: {np.ascontiguousarray(data)}")

        cuda.memcpy_htod_async(self.d_input, np.ascontiguousarray(data), self.stream)
        self.stream.synchronize()
        # print(f'PyCUDA TensorRT copy h2d and synchronization takes: {(time.time() - start) * 1000} ms')

        if self.debug:
            start = time.time()

        self.context.execute_async_v2([int(self.d_input), int(self.d_output)], self.stream.handle, None)
        # self.context.execute_async_v2(stream_handle=stream.handle)
        if self.debug:
            end = time.time()
            print(f'PyCUDA TensorRT inference takes: {(end - start) * 1000} ms')
        output = np.empty(self.output_shape, dtype=np.float32)

        # start = time.time()
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        # print(f'PyCUDA TensorRT copy d2h and synchronization takes: {(time.time() - start) * 1000} ms')

        if self.enable_nms:
            return self.postprocess_output(output)
        return output

    @staticmethod
    def _to_xyxy(xywh):
        boxes = np.empty_like(xywh, dtype=np.float32)
        half_w = xywh[:, 2] * 0.5
        half_h = xywh[:, 3] * 0.5
        boxes[:, 0] = xywh[:, 0] - half_w
        boxes[:, 1] = xywh[:, 1] - half_h
        boxes[:, 2] = xywh[:, 0] + half_w
        boxes[:, 3] = xywh[:, 1] + half_h
        return boxes

    @staticmethod
    def _fast_nms_single_class(boxes, scores, iou_thres=0.45, max_det=300):
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0 and len(keep) < max_det:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            union = np.maximum(area_i + area_rest - inter, 1e-9)
            iou = inter / union

            order = rest[iou <= iou_thres]

        return np.asarray(keep, dtype=np.int64)

    @func_timer
    def postprocess_output(self, output):
        pred = np.asarray(output, dtype=np.float32)

        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]

        # Accept both [5, 8400] and [8400, 5].
        if pred.ndim != 2:
            raise ValueError(f"Unexpected output ndim={pred.ndim}, expected 2D tensor")
        if pred.shape[0] == 5:
            pred = pred.T
        elif pred.shape[1] != 5:
            raise ValueError(f"Unexpected output shape={pred.shape}, expected (5,N) or (N,5)")

        scores = pred[:, 4]
        conf_mask = scores >= self.conf_thres
        if not np.any(conf_mask):
            return np.empty((0, 5), dtype=np.float32)

        cand = pred[conf_mask]
        boxes = self._to_xyxy(cand[:, :4])
        scores = cand[:, 4]

        # Top-k prefilter before NMS to reduce sorting/IoU cost.
        if self.pre_nms_topk > 0 and scores.size > self.pre_nms_topk:
            top_idx = np.argpartition(scores, -self.pre_nms_topk)[-self.pre_nms_topk:]
            boxes = boxes[top_idx]
            scores = scores[top_idx]

        keep = self._fast_nms_single_class(
            boxes,
            scores,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
        )

        out = np.concatenate([boxes[keep], scores[keep, None]], axis=1)

        if self.debug:
            print(f'NMS keep: {out.shape[0]}')

        return out

if __name__ == '__main__':
    import cv2
    def draw_bbox_xyxy(image, boxes_xyxy, color=(0, 0, 255), thickness=2, category="drone"):
        """Simplified drawer: input np image + xyxy boxes, return rendered image."""
        """
            o -- x
            |
            y
        """
        if image is None:
            raise ValueError("image is None")

        out = image.copy()
        h, w = out.shape[:2]
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)

        # Support one label for all boxes, or one label per box.
        if isinstance(category, (list, tuple, np.ndarray)):
            labels = [str(x) for x in category]
        else:
            labels = [str(category)] * len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (box[:] * max(h, w)).astype(np.int32)
            # x1, x2 = (box[0] * w).astype(np.int32), (box[2] * w).astype(np.int32)
            # y1, y2 = (box[0] * h).astype(np.int32), (box[2] * h).astype(np.int32)
            # print(w, h)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            if i < len(labels) and labels[i] != "":
                cv2.putText(
                    out,
                    labels[i],
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )

        return out
    # det = CUDADetector("./stage1_half_fp16.engine", debug=True)
    det = CUDADetector(
        "./stage1.engine",
        debug=True,
        enable_nms=True,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        pre_nms_topk=1000,
    )
    canvas = np.zeros([1, 3, 640, 640], dtype=np.float32)
    img = cv2.imread("./test_data/i_0.png", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_raw = cv2.resize(img, (640, 480))
    img = (cv2.resize(img, (640, 480)) / 255.0).astype(np.float32)
    img = img.transpose(2, 0, 1)[None]
    canvas[:, :, :480, :640] = img
    img = canvas
    print(img.shape)
    ret = det.detect(img).squeeze()
    print(ret.shape)
    print(np.all(ret == 0))
    print(ret)

    bbox = ret[:4] / 640

    output = draw_bbox_xyxy(img_raw, bbox)

    cv2.imwrite("./test_data/output.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

