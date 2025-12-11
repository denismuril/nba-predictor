import os
import re
import logging
import sys
from pathlib import Path

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

def check_hardcoded_secrets():
    """Varre o código em busca de chaves de API hardcoded."""
    logger.info("🕵️‍♂️ Iniciando varredura de segredos hardcoded...")
    
    suspicious_patterns = [
        r"AIza[0-9A-Za-z-_]{35}", # Google API Key
        r"AAAAAAAAAAAAAAAAAAAAA", # Twitter Bearer Token (Parcial)
        r"sk-[a-zA-Z0-9]{48}",    # OpenAI/Stripe etc
        r"ODDS_API_KEY\s*=\s*['\"][a-zA-Z0-9]{10,}['\"]", # Nossa chave de odds
    ]
    
    files_to_scan = list(PROJECT_ROOT.glob("**/*.py"))
    issues_found = 0
    
    for file_path in files_to_scan:
        if "venv" in str(file_path) or "security_audit.py" in str(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in suspicious_patterns:
                    if re.search(pattern, content):
                        # Ignorar placeholders conhecidos
                        if "SUA_CHAVE_AQUI" in content:
                            continue
                            
                        logger.warning(f"⚠️  Possível segredo encontrado em: {file_path.name}")
                        issues_found += 1
        except Exception:
            pass
            
    if issues_found == 0:
        logger.info("✅ Nenhum segredo óbvio encontrado no código.")
    else:
        logger.warning(f"⚠️  {issues_found} arquivos com possíveis segredos hardcoded. Mova para .env!")

def check_file_permissions():
    """Verifica permissões de arquivos sensíveis."""
    logger.info("🔒 Verificando permissões de arquivos sensíveis...")
    
    sensitive_files = ['.env', 'config/constants.py']
    
    for filename in sensitive_files:
        path = PROJECT_ROOT / filename
        if path.exists():
            # Em Linux/WSL, queremos 600 (apenas dono lê)
            try:
                mode = os.stat(path).st_mode
                # Converter para octal (ex: 0o100644)
                perm = oct(mode)[-3:]
                
                if perm != '600':
                    logger.warning(f"⚠️  Permissão insegura em {filename}: {perm}. Recomendado: 600")
                else:
                    logger.info(f"✅ Permissões de {filename} seguras (600).")
            except Exception as e:
                logger.error(f"Erro ao verificar permissões de {filename}: {e}")

def check_dependencies():
    """Verifica vulnerabilidades conhecidas (Simulado/Básico)."""
    logger.info("📦 Auditando dependências...")
    
    # Em produção, usaríamos 'safety check' ou 'pip-audit'
    try:
        import pkg_resources
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        
        # Lista de versões vulneráveis conhecidas (Exemplo)
        vulnerable = {
            'pandas': ['0.25.0'],
            'numpy': ['1.16.0'],
            'requests': ['2.19.0']
        }
        
        issues = 0
        for pkg, bad_versions in vulnerable.items():
            if pkg in installed:
                if installed[pkg] in bad_versions:
                    logger.warning(f"🚨 Dependência VULNERÁVEL detectada: {pkg} {installed[pkg]}")
                    issues += 1
                    
        if issues == 0:
            logger.info("✅ Nenhuma vulnerabilidade crítica conhecida nas libs principais.")
            
    except Exception as e:
        logger.error(f"Erro ao auditar dependências: {e}")

def run_security_audit():
    logger.info("🛡️  INICIANDO SECURITY AUDIT (PENTEST AUTOMATIZADO) 🛡️")
    logger.info("="*60)
    
    check_hardcoded_secrets()
    print("-" * 30)
    check_file_permissions()
    print("-" * 30)
    check_dependencies()
    
    logger.info("="*60)
    logger.info("🏁 Auditoria finalizada.")

if __name__ == "__main__":
    run_security_audit()
