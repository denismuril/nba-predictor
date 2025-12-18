#!/usr/bin/env python3
"""
Teste Manual Rápido - Player Props

Este script testa manualmente o scraping de Player Props
do Action Network em modo visual (headless=False).

Execute: python scripts/test_player_props_manual.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Adiciona raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_player_props_visual():
    """Testa scraping de props visualmente."""
    print("\n" + "=" * 70)
    print("  TESTE VISUAL: Action Network Player Props")
    print("=" * 70)
    
    from data.scrapers.action_network_scraper import ActionNetworkScraper
    
    # Modo NON-headless para ver o que está acontecendo
    print("\n🌐 Iniciando navegador (você verá a janela abrir)...")
    print("⏱️  Aguarde ~10-15 segundos para o scraping completar...\n")
    
    scraper = ActionNetworkScraper(headless=False)
    
    try:
        # Data de hoje
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"📅 Buscando props para: {today}\n")
        
        props = await scraper.fetch_props(today)
        
        print("\n" + "=" * 70)
        print(f"  RESULTADOS: {len(props)} player props encontrados")
        print("=" * 70 + "\n")
        
        if props:
            # Mostra primeiros 10
            print("📊 Primeiros 10 props:\n")
            for i, prop in enumerate(props[:10], 1):
                print(f"{i:2d}. {prop.player_name:20s} | {prop.prop_type:8s} | "
                      f"Line: {prop.line:5.1f} | Over: {prop.over_odds:.3f} | "
                      f"Under: {prop.under_odds:.3f}")
            
            # Estatísticas
            from collections import Counter
            by_type = Counter(p.prop_type for p in props)
            
            print("\n📈 Distribuição por tipo de prop:\n")
            for prop_type, count in by_type.most_common():
                bar = "█" * (count // 5)
                print(f"  {prop_type:12s}: {count:3d} {bar}")
            
            print("\n✅ SUCESSO! Player props funcionando perfeitamente.")
            
        else:
            print("⚠️  Nenhum prop encontrado. Possíveis razões:")
            print("   - Não há jogos hoje")
            print("   - Site bloqueou o scraping")
            print("   - Seletores/API mudaram")
            
    except Exception as e:
        print(f"\n❌ ERRO durante scraping:")
        print(f"   {type(e).__name__}: {e}")
        print("\n💡 Dica: Verifique se Playwright está instalado:")
        print("   playwright install chromium")
        import traceback
        traceback.print_exc()


async def test_with_odds_manager():
    """Testa via OddsDataManager (método de produção)."""
    print("\n" + "=" * 70)
    print("  TESTE: OddsDataManager.fetch_player_props()")
    print("=" * 70 + "\n")
    
    from data.odds_manager import OddsDataManager
    
    manager = OddsDataManager()
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📅 Buscando via OddsDataManager para: {today}\n")
    
    try:
        props = await manager.fetch_player_props(today)
        
        print(f"\n✅ Manager retornou {len(props)} props")
        
        if props:
            print(f"\nExemplo: {props[0].player_name} - {props[0].prop_type} {props[0].line}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


async def main():
    """Menu de teste."""
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "TESTE MANUAL - PLAYER PROPS SCRAPER" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    print("Escolha o teste:\n")
    print("1. Teste Visual (headless=False) - Recomendado para primeira vez")
    print("2. Teste via OddsDataManager (headless=True)")
    print("3. Ambos")
    print()
    
    choice = input("Opção [1-3]: ").strip()
    
    if choice == "1":
        await test_player_props_visual()
    elif choice == "2":
        await test_with_odds_manager()
    elif choice == "3":
        await test_player_props_visual()
        await test_with_odds_manager()
    else:
        print("Opção inválida. Execute novamente.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Teste cancelado pelo usuário.")
