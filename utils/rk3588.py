from rknnlite.api import RKNNLite


def create_rknn_session(
    model_path: str, core_mask: int = RKNNLite.NPU_CORE_0
) -> RKNNLite:
    rknn_lite = RKNNLite()

    ret = rknn_lite.load_rknn(model_path)
    if ret:
        raise OSError(f"{model_path}: Export rknn model failed!")

    ret = rknn_lite.init_runtime(async_mode=True, core_mask=core_mask)
    if ret:
        raise OSError(f"{model_path}: Init runtime enviroment failed!")

    return rknn_lite
