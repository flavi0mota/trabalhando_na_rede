# server_raw.py
import socket
import struct
import json
import joblib
import numpy as np

# Carrega o modelo (ex.: Random Forest treinado)
modelo = joblib.load('rf_model.pkl')

HOST = '0.0.0.0'
PORT = 51515

def recvall(conn, n):
    """Recebe exatamente n bytes."""
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def handle_client(conn, addr):
    print(f"Conexão de {addr}")
    try:
        raw_len = recvall(conn, 4)
        if not raw_len: return
        msg_len = struct.unpack('>I', raw_len)[0]
        payload = recvall(conn, msg_len)
        if not payload: return
        request = json.loads(payload.decode('utf-8'))
        features = request['features']

        # DEBUG: mostre as features
        print(f"Features recebidas: {features}")

        entrada = np.array(features).reshape(1, -1)
        pred = modelo.predict(entrada)[0]

        # DEBUG: mostre a previsão e probabilidades
        print(f"Previsão: {pred}")
        if hasattr(modelo, 'predict_proba'):
            proba = modelo.predict_proba(entrada)
            print(f"Probabilidades: {proba}")

        resposta = json.dumps({'prediction': int(pred)})
        resp_bytes = resposta.encode('utf-8')
        conn.sendall(struct.pack('>I', len(resp_bytes)) + resp_bytes)
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        conn.close()

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Servidor ouvindo em {HOST}:{PORT}...")
        while True:
            conn, addr = s.accept()
            # Simplicidade: um cliente por vez. Para concorrência, usar threads ou select.
            handle_client(conn, addr)

if __name__ == '__main__':
    main()
