"""
Data Ingestion Module - NBA Stats via nba_api

Este módulo fornece clientes modernos para ingestão de dados da NBA.
Substitui scrapers antigos de stats por chamadas oficiais à API.

v27.0: Implementação inicial.
"""

from .stats_client import NBAStatsClient

__all__ = ['NBAStatsClient']
