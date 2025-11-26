# 📋 Checklist ErgoPose Risk Classifier

## **Modelos e Desempenho**
- [ ] **Salvar modelos treinados**:
  - Salvar os modelos após o treinamento para poder compará-los e utilizá-los posteriormente.
  - Gerar um histórico de desempenho com as métricas relevantes (ex: acurácia, precisão, recall, F1-Score).
  - Utilizar `model.save()` ou bibliotecas como `joblib`/`pickle` para armazenar os modelos.
  
- [ ] **Gerar histogramas a partir do histórico de desempenho**:
  - Criar gráficos de desempenho dos modelos (ex: acurácia por época) para facilitar a comparação e visualização dos resultados.
  - Utilizar `matplotlib` ou `seaborn` para criar histogramas e gráficos de barras.

## **Dependências e Compatibilidade**
- [ ] **Verificar versão do PyTorch no `requirements.txt`**:
  - Verificar se a versão do PyTorch no `requirements.txt` é compatível com a GPU disponível.
  - Ajustar a versão conforme necessário para otimizar o desempenho, especialmente se estiver utilizando CUDA.

## **Documentação**
- [ ] **Atualizar o `README.md`**:
  - Incluir informações detalhadas sobre a arquitetura do modelo, como a configuração das redes neurais e a escolha de hiperparâmetros.
  - Detalhar o processo de pré-processamento dos dados e a metodologia utilizada.
  - Incluir informações sobre como reproduzir os experimentos e executar o código.
  - Adicionar um link para o dataset (Zenodo) e como usá-lo no projeto.


# 📊 Estrutura da Apresentação - "ErgoPose Risk Classifier"

## 1. **Sobre o Projeto**
- **Objetivo**: Classificar posturas ergonômicas para avaliação de risco usando redes neurais.
- **Dataset**: MultiPosture - Contém coordenadas esqueléticas extraídas de vídeos de participantes em diferentes posturas sentadas.
- **Desafio**: Melhorar a detecção de posturas para prevenir distúrbios musculoesqueléticos.

## 2. **Pipeline**
- **Coleta de Dados**: Obtenção do dataset (Zenodo).
- **Pré-processamento**: Remoção da coordenada Z, normalização, e engenharia de features (ângulos de postura).
- **Treinamento**: Utilização de redes neurais para classificação das posturas.
- **Validação**: 5-Fold Cross-Validation para avaliação do modelo.
- **Resultados**: Comparação entre diferentes modelos e análise de desempenho.

## 3. **Overview**
- **Histograma de Modelos**: 
  - Gráfico comparativo da **acurácia** de todos os modelos treinados.
  - Visualização do desempenho ao longo das épocas de treinamento.
  - Comparação entre modelos para avaliar qual se destaca em termos de acurácia e outras métricas.

## 4. **Pior Modelo**
- **Curva de Aprendizado**: 
  - Exibição da curva de aprendizado do pior modelo, mostrando a evolução da acurácia e perda ao longo do tempo.
- **Acurácia**:
  - Acurácia final do modelo com o pior desempenho.
- **Métricas**: 
  - Precision, Recall e F1-Score para entender as limitações do modelo.

## 5. **Top 3 Melhores Modelos e Seus Resultados**
- **Modelo 1 (Melhor desempenho)**:
  - Resultados detalhados: Acurácia, Precision, Recall, F1-Score.
- **Modelo 2 (2º Melhor desempenho)**:
  - Resultados detalhados: Acurácia, Precision, Recall, F1-Score.
- **Modelo 3 (3º Melhor desempenho)**:
  - Resultados detalhados: Acurácia, Precision, Recall, F1-Score.
  
  - **Gráficos Comparativos**: 
    - Performance visual dos três melhores modelos em termos de aprendizado e métricas.

## 6. **Considerações Finais**
- **Insights**: O que aprendemos com os resultados obtidos, como os diferentes modelos se comportaram.
- **Desafios**: Limitações e obstáculos enfrentados durante o treinamento e avaliação dos modelos.
- **Próximos Passos**: Possíveis melhorias no modelo, novos experimentos, e o uso do sistema em tempo real para avaliação de risco postural.
