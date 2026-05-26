#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>
#include <cmath>
#include <random>
#include "simple_json.hpp"

constexpr uint32_t MAX_RESPONSE_SIZE = 10240;

bool recv_all(int sock, void* buffer, size_t length) {
    char* ptr = static_cast<char*>(buffer);
    size_t received = 0;
    while (received < length) {
        ssize_t r = recv(sock, ptr + received, length - received, 0);
        if (r <= 0) {
            return false;
        }
        received += r;
    }
    return true;
}

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Erro ao criar socket\n";
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(51515);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Erro na conexão com o servidor. O servidor Python está a correr?\n";
        return 1;
    }
    std::cout << "Conectado ao servidor\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<double> dist_t(0.0, 4.0 * M_PI);
    
    std::normal_distribution<double> dist_noise(0.0, 0.3);

    std::vector<double> features(5);
    
    double t_start = dist_t(gen);
    double step = 0.1;
    
    std::cout << "Gerando features aleatórias a partir de t = " << t_start << "...\n";
    for(int i = 0; i < 5; i++) {
        features[i] = std::sin(t_start + i * step) + dist_noise(gen);
    }

    std::string json_str = SimpleJson::build_request(features);
    std::cout << "Enviando: " << json_str << std::endl;

    uint32_t msg_len = htonl(json_str.size());
    send(sock, &msg_len, sizeof(msg_len), 0);
    send(sock, json_str.c_str(), json_str.size(), 0);

    uint32_t resp_len_net;
    if (!recv_all(sock, &resp_len_net, sizeof(resp_len_net))) {
        std::cerr << "Erro ao receber tamanho do payload\n";
        close(sock);
        return 1;
    }
    uint32_t resp_len = ntohl(resp_len_net);

    if (resp_len == 0 || resp_len > MAX_RESPONSE_SIZE) {
        std::cerr << "Tamanho de resposta inválido ou excessivamente grande (" << resp_len << " bytes).\n";
        close(sock);
        return 1;
    }

    std::vector<char> resp_buffer(resp_len);
    if (!recv_all(sock, resp_buffer.data(), resp_len)) {
        std::cerr << "Erro ao receber payload\n";
        close(sock);
        return 1;
    }
    std::string resp_str(resp_buffer.data(), resp_len);
    std::cout << "Resposta bruta: " << resp_str << std::endl;

    if (resp_str.find("\"error\"") != std::string::npos) {
        std::cerr << "O servidor retornou um erro estrutural!\n";
    } else {
        try {
            int prediction = SimpleJson::parse_prediction(resp_str);
            std::cout << "Previsão do modelo: " << prediction << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Erro ao interpretar JSON: " << e.what() << std::endl;
        }
    }

    close(sock);
    return 0;
}
