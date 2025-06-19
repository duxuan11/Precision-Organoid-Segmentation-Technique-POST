# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.



def import_model_class(model_type, class_name):
    """
    Imports a predefined detection class by class name.

    Args:
        model_type: str
            "yolov5", "detectron2", "mmdet", "huggingface" etc
        model_name: str
            Name of the detection model class (example: "MmdetDetectionModel")
    Returns:
        class_: class with given path
    """
    module = __import__(f"POST.models.detector.{model_type}", fromlist=[class_name]) #动态方式导入模块
    class_ = getattr(module, class_name)   #getattr(t, "run")() #获取run方法，后面加括号可以将这个方法运行。
    return class_


