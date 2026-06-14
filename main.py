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
    logger.info("Metricas out-of-time: %s", result.out_of_time_metrics)
    logger.info("Decisao de baseline: %s", result.baseline_decision)
    if settings.storage_backend == "minio":
        logger.info("Pipeline salva em: %s", settings.object_uri(settings.pipeline_object_key))
        logger.info(
            "Analise de thresholds salva em: %s",
            settings.object_uri(
                settings.artifact_object_key(settings.threshold_analysis_filename)
            ),
        )
        logger.info(
            "Auditoria de leakage salva em: %s",
            settings.object_uri(
                settings.artifact_object_key(settings.leakage_report_filename)
            ),
        )
        logger.info(
            "Historico do treino salvo | run_id=%s | diretorio=%s",
            result.run_id,
            settings.object_uri(settings.artifact_object_key(f"history/{result.run_id}")),
        )
    else:
        logger.info("Pipeline salva em: %s", result.pipeline_path)
        logger.info("Analise de thresholds salva em: %s", result.threshold_analysis_path)
        logger.info("Auditoria de leakage salva em: %s", result.leakage_report_path)
        logger.info(
            "Historico do treino salvo | run_id=%s | diretorio=%s",
            result.run_id,
            result.history_run_dir,
        )


if __name__ == "__main__":
    main()
