# Preprocessing module
from .face_extractor import FaceExtractor, RetinaFaceExtractor
from .frame_sampler import FrameSampler
from .transforms import get_train_transforms, get_val_transforms
from .pipeline import PreprocessingPipeline
from .preprocess_dataset import DatasetPreprocessor, create_preprocessor_from_config
from .preprocess_celeb_df import CelebDFPreprocessor

__all__ = [
    'FaceExtractor',
    'RetinaFaceExtractor', 
    'FrameSampler',
    'get_train_transforms',
    'get_val_transforms',
    'PreprocessingPipeline',
    'create_pipeline_from_config',
    'DatasetPreprocessor',
    'create_preprocessor_from_config',
    'CelebDFPreprocessor'
]

def create_pipeline_from_config(config):
    """Create a PreprocessingPipeline from a configuration object."""
    
    return PreprocessingPipeline(
        num_frames=config.preprocessing.frames_per_video,
        sampling_strategy=config.preprocessing.frame_sampling_strategy,
        face_detector=config.preprocessing.face_detector,
        detection_threshold=config.preprocessing.face_detection_threshold,
        output_size=config.preprocessing.output_size,
        bbox_enlargement=config.preprocessing.bbox_enlargement_factor,
        align_faces=True,
        device=config.training.device,
        seed=config.seed
    )
