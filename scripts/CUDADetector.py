from detect_utils import DetectResult
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import time
def func_timer(func, tag='Detector'):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'{tag}: function {func.__name__} takes {(time.perf_counter() - start) * 1000:.2f}ms')
        return result
    return wrapper

class CUDADetector:
    """
        This is CUDA Detector class for TensorRT engine, which generally suitable for
        YOLOxx-pose. The engine file should be exported via NVIDIA TensorRT API : trtexec.
        The input data should be normalized to 1 and in BCHW dimension order.
        The output data shape is (1, 300, 57), i.e. (B, N, I), where B for the batch size,
        N for the number of maximum detections, I for the dimension of information.

        The output 57 dimension I is like:
            origin: left-top:
            o -- x
            |
            y
            0-1: x1, y1 -> the absolute left-top coordinates of the bbox
            2-3: x2, y2 -> the absolute right-bottom coordinates of the bbox
            4: obj_conf -> the confidence of the detection result ∈ (0, 1)
            5: cls -> class index of the detected object, 0 for person
            6-56: (<x0, y0, v0>, <x1, y1, v1>, ..., <x16, y16, v16>, ) ->
            absolute x, y position of 17 key points and their visibility confidence
    """
    def __init__(self, engine_path, debug=False):
        self.debug = debug
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

        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize))
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize))

        self.stream = cuda.Stream()

    def detect(self, data):
        # start = time.time()
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

        return self.postprocess_output(output.squeeze())

    # @func_timer
    def postprocess_output(self, output):
        conf_mask = output[:, 4] > 0.75
        cls_mask = (output[:, 5]) < 1e-4
        filtered_results = output[np.logical_and(cls_mask, conf_mask), ...]
        if filtered_results.shape[0] == 0:
            return None

        # x1 = filtered_results[..., 0]
        # y1 = filtered_results[..., 1]
        # x2 = filtered_results[..., 2]
        # y2 = filtered_results[..., 3]
        # print(type(filtered_results))
        xyxy = filtered_results[..., :4]
        # x1, y1, x2, y2 = filtered_results[..., :4]
        conf = filtered_results[..., 4]
        key_points = filtered_results[..., 6:].reshape(-1, 17, 3)

        ret = DetectResult(xyxy, conf, key_points)
        # print(type(xyxy))
        if self.debug:
            print(f'bbox: {xyxy}')

        return ret
