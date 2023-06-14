# Cálculo IDF - Python

Este repositório contém o código Python para calcular a Intensidade-Duração-Frequência (IDF) a partir de dados de chuva. O código lê um arquivo CSV com dados de chuva, processa os dados e calcula a IDF.

## Contexto

A Intensidade-Duração-Frequência (IDF) é uma medida importante em hidrologia e é usada para planejar e projetar infraestruturas de drenagem. Este projeto visa fornecer uma ferramenta para calcular a IDF a partir de dados de chuva.

## Funcionalidades

- Ler um arquivo CSV com dados de chuva.
- Processar os dados para remover outliers e calcular a média mensal.
- Calcular a IDF usando a fórmula de Ven Te Chow.

## Tecnologias Utilizadas

- Python para a implementação do código.
- Pandas para a manipulação de dados.
- Numpy para cálculos numéricos.

## Como Usar

1. Clone este repositório para a sua máquina local.
2. Instale as dependências do projeto com o comando `pip install -r requirements.txt`.
3. Execute o código com o comando `python src/main.py`.

## Estrutura do Projeto

- `src/main.py`: Este é o arquivo principal do projeto. Ele lê o arquivo CSV, processa os dados e calcula a IDF.
- `src/process_data.py`: Este arquivo contém a função para processar os dados do arquivo CSV.
- `src/yn_sigman.py`: Este arquivo contém um dicionário que mapeia valores de 'n' para 'yn_sigman'.

## Contribuindo

Contribuições são sempre bem-vindas! Sinta-se à vontade para abrir uma issue ou fazer um pull request.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
