import sys
import os
from datetime import datetime

class Tee:
    """
    Redireciona o stdout para múltiplos arquivos (ex: terminal e um arquivo de log).
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def setup_logging(log_dir):
    """
    Configura o logging para redirecionar o stdout para um arquivo e para o console.

    Args:
        log_dir (str): O diretório onde o arquivo de log será salvo.

    Returns:
        tuple: Uma tupla contendo o stdout original, o objeto do arquivo de log,
               e o caminho para o arquivo de log.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file_path = os.path.join(log_dir, f"main_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    original_stdout = sys.stdout
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = Tee(original_stdout, log_file)
    
    print(f"Log de execução será salvo em: {log_file_path}\n")
    
    return original_stdout, log_file, log_file_path

def close_logging(original_stdout, log_file, log_file_path):
    """
    Restaura o stdout e fecha o arquivo de log.

    Args:
        original_stdout: O objeto sys.stdout original.
        log_file: O objeto do arquivo de log a ser fechado.
        log_file_path (str): O caminho para o arquivo de log para a mensagem final.
    """
    sys.stdout = original_stdout
    log_file.close()
    print(f"\nLog completo salvo em: {log_file_path}") 