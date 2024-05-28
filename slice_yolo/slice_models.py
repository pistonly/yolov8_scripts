from ultralytics import YOLOWorld
from ultralytics.nn.tasks import WorldModel
from ultralytics.models import yolo
from .slice_val import SliceDetectionValidator


class YOLOWorldSlice(YOLOWorld):

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": SliceDetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }
