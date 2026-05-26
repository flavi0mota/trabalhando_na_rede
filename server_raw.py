#!/usr/bin/env python3
"""
Servidor TCP bruto (socket + threading) para predição com Random Forest.
Protocolo:
  - Recebe 4 bytes (big-endian) com tamanho do JSON.
  - JSON: {"features": [f1, f2, ...]}
  - Responde 4 bytes + JSON {"prediction": int} ou {"error": "..."}
"""

import socket
import struct
import json
import threading
import logging
import signal
import sys
import numpy as np
import joblib

# ------------------------------------------------------------
# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Variável global para o modelo (carregada no main)
modelo = None


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Recebe exatamente n bytes de um socket, ou levanta exceção."""
    data = b''
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except ConnectionResetError:
            raise ConnectionError("Conexão resetada pelo cliente")
        if not chunk:
            raise ConnectionError("Conexão fechada pelo cliente")
        data += chunk
    return data


def handle_client(conn: socket.socket, addr: tuple):
    """Thread de atendimento a um cliente."""
    logger.info(f"Nova conexão de {addr}")
    try:
        # 1. Lê o tamanho da mensagem (4 bytes big-endian)
        raw_len = recv_exact(conn, 4)
        msg_len = struct.unpack('>I', raw_len)[0]

        # 2. Lê o payload JSON
        payload = recv_exact(conn, msg_len)
        request = json.loads(payload.decode('utf-8'))
        features = request['features']
        logger.info(f"Features recebidas de {addr}: {features}")

        # 3. Predição
        entrada = np.array(features, dtype=np.float32).reshape(1, -1)
        pred = modelo.predict(entrada)[0]

        # 4. Probabilidades (se disponível)
        proba = None
        if hasattr(modelo, 'predict_proba'):
            proba = modelo.predict_proba(entrada)[0].tolist()
            logger.info(f"Predição={pred}, Probabilidades={proba}")

        # 5. Monta e envia resposta
        resposta = json.dumps({'prediction': int(pred)})
        resp_bytes = resposta.encode('utf-8')
        conn.sendall(struct.pack('>I', len(resp_bytes)) + resp_bytes)
        logger.info(f"Resposta enviada para {addr}")

    except Exception as e:
        logger.error(f"Erro ao processar requisição de {addr}: {e}", exc_info=True)
        # Tenta enviar uma resposta de erro (se a conexão ainda estiver viva)
        try:
            erro_msg = json.dumps({'error': str(e)}).encode('utf-8')
            conn.sendall(struct.pack('>I', len(erro_msg)) + erro_msg)
        except:
            pass
    finally:
        conn.close()
        logger.info(f"Conexão com {addr} encerrada")


def main():
    global modelo

    # Carrega o modelo uma única vez
    modelo = joblib.load('rf_model.pkl')
    logger.info("Modelo carregado com sucesso.")

    HOST = '0.0.0.0'
    PORT = 51515

    # Cria o socket servidor
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        # Permite reutilizar o endereço imediatamente
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_sock.bind((HOST, PORT))
        server_sock.listen(5)
        logger.info(f"Servidor ouvindo em {HOST}:{PORT} (socket + threading)")

        # Tratamento gracioso de SIGINT (Ctrl+C)
        shutdown_event = threading.Event()

        def signal_handler(sig, frame):
            logger.info("Sinal de encerramento recebido. Aguardando threads...")
            shutdown_event.set()
            server_sock.close()  # força saída do accept
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while not shutdown_event.is_set():
            try:
                conn, addr = server_sock.accept()
                # Cria uma thread daemon para o cliente
                t = threading.Thread(target=handle_client, args=(conn, addr))
                t.daemon = True
                t.start()
            except socket.error:
                # Se o socket for fechado pelo signal handler, o accept falha
                if shutdown_event.is_set():
                    break
                logger.error("Erro no accept", exc_info=True)

        logger.info("Servidor finalizado.")


if __name__ == '__main__':
    main()
