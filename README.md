# Cliente‑Servidor de Previsão de Fluxo de Rede com Random Forest

Projeto acadêmico que implementa uma comunicação **cliente‑servidor pura via sockets TCP** entre um cliente escrito em C++ e um servidor Python que executa um modelo de Random Forest para previsão de fluxo de rede.  
O foco é demonstrar a criação “visceral” do protocolo de comunicação, incluindo construção manual de JSON no cliente e parsing próprio, sem bibliotecas externas de rede ou serialização.

---

## 📁 Estrutura do Projeto

```
.
├── README.md
├── train_model.py          # Script para gerar o modelo Random Forest (.pkl)
├── server_raw.py           # Servidor Python (socket + scikit‑learn)
├── simple_json.hpp         # Biblioteca C++ para JSON artesanal
├── main.cpp                # Cliente C++ (socket + JSON próprio)
├── execute_server.sh       # Script Bash para iniciar o servidor
└── execute_client.sh       # Script Bash para compilar e executar o cliente
```

---

## 🧠 Pré‑requisitos

**Servidor (Python 3.6+)**
```bash
pip install scikit-learn joblib numpy
```

**Cliente (C++17)**
- Compilador `g++` (Linux) ou `clang++`
- Bibliotecas padrão do sistema (sem dependências externas)

---

## 🚀 Como executar

### 1. Treinar o modelo (gerar `rf_model.pkl`)
```bash
python train_model.py
```
Isso criará o arquivo `rf_model.pkl` com o classificador treinado.

### 2. Iniciar o servidor
```bash
bash execute_server.sh
```
ou diretamente:
```bash
python server_raw.py
```
O servidor escuta em `0.0.0.0:51515`.

### 3. Executar o cliente
Em outro terminal, execute:
```bash
bash execute_client.sh
```
Esse script compila `main.cpp` (se necessário) e roda o cliente, que envia features de exemplo e exibe a previsão retornada.

---

## 🔧 Scripts auxiliares

### `execute_server.sh`
```bash
#!/bin/bash
echo "Iniciando servidor Python..."
python server_raw.py
```

### `execute_client.sh`
```bash
#!/bin/bash
echo "Compilando cliente C++..."
g++ -std=c++17 main.cpp -o client_raw
if [ $? -eq 0 ]; then
    echo "Executando cliente..."
    ./client_raw
else
    echo "Erro na compilação."
fi
```

---

1. **Cliente C++** coleta features (ex.: estatísticas de tráfego) e monta um JSON simples usando `simple_json.hpp`.
2. O JSON é enviado via socket TCP precedido por um cabeçalho de 4 bytes com o tamanho do payload (big‑endian).
3. **Servidor Python** recebe, decodifica, aplica o modelo Random Forest e responde com outro JSON (`{"prediction": valor}`).
4. O cliente extrai o valor da previsão usando o parser próprio e o exibe.

---

O vetor de features enviado pelo cliente está definido no `main.cpp`:
```cpp
std::vector<double> features = {80.5, 40.2, 50, 30, 70};
```
Altere esses valores para simular diferentes cenários e ver a resposta mudar entre `0` e `1`.

---

## 🎓 Contexto acadêmico

Trabalho de redes com ênfase em:
- Criação manual de sockets (C++ e Python)
- Serialização/desserialização de mensagens (JSON artesanal)
- Integração com modelo de Machine Learning
- Observação do tráfego real com ferramentas como Wireshark

---

## 📄 Licença

Projeto para fins educacionais. Sinta‑se livre para adaptar e compartilhar.
```