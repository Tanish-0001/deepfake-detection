# Preprocessing module
from .face_extractor import FaceExtractor, RetinaFaceExtractor
from .frame_sampler import FrameSampler
from .transforms import get_train_transforms, get_val_transforms
from .pipeline import PreprocessingPipeline
from .preprocess_dataset import DatasetPreprocessor, create_preprocessor_from_config

__all__ = [
    'FaceExtractor',
    'RetinaFaceExtractor', 
    'FrameSampler',
    'get_train_transforms',
    'get_val_transforms',
    'PreprocessingPipeline',
    'create_pipeline_from_config',
    'DatasetPreprocessor',
    'create_preprocessor_from_config'
]

def create_pipeline_from_config(config):
    """Create a PreprocessingPipeline from a configuration object."""
    
    return PreprocessingPipeline(
        num_frames=config.data.frames_per_video,
        sampling_strategy=config.data.frame_sampling_strategy,
        face_detector=config.data.face_detector,
        detection_threshold=config.data.face_detection_threshold,
        output_size=config.data.output_size,
        bbox_enlargement=config.data.bbox_enlargement_factor,
        align_faces=True,
        device=config.training.device,
        seed=config.seed
    )
