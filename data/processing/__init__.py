"""
Data Processing Module - ETL Pipeline para Player Props.

Este módulo contém o motor de integração que processa dados brutos dos scrapers
e gera features contextuais para inferência do modelo XGBoost.

v27.0: Implementação inicial do pipeline God Mode.
"""

from .props_processor import PropsProcessor, PropsProcessorConfig

__all__ = ["PropsProcessor", "PropsProcessorConfig"]
