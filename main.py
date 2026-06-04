"""Entry point for training the fraud detection pipeline."""

from src.config.settings import Settings
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    """Train, evaluate, tune the threshold and persist the complete pipeline."""
    settings = Settings()
    result = TrainingPipeline(settings).run()
    logger.info("Modelo treinado: %s", result.model_name)
    logger.info("Threshold selecionado: %.4f", result.threshold)
    logger.info("Metricas de validacao: %s", result.validation_metrics)
    logger.info("Metricas de teste: %s", result.test_metrics)
    logger.info("Pipeline salva em: %s", result.pipeline_path)


if __name__ == "__main__":
    main()
