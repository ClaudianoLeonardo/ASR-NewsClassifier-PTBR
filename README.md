# Automatic Speech Recognition News Classifier - PTBR
Esse projeto consiste em uma arquitetura composta por dois modelos,Whisper e Bert, para realizar classificação de noticias em portugues. Para desempenhar essa atividade foi realizado o fine tuning do Bert utilzando a base de dados [noticias publicadas no Brasil](https://www.kaggle.com/datasets/diogocaliman/notcias-publicadas-no-brasil) e feito o tracking das métricas de treinamento utilizando o mlflow. Para integração dos dois modelos foi desenvolvido uma interface gráfica utilizando gradio. O deploy do modelo está disponível em um hugging face space.

## Conjunto de dados:
A base de dados conta com 10 mil noticias divididas em 5 classes: esportes, economia, famosos, politica, tecnologia.

para mais informações acesse o [link](https://www.kaggle.com/datasets/diogocaliman/notcias-publicadas-no-brasil).

## MLflow:

MLflow é uma plataforma de código aberto para gerenciamento de ciclo de vida de projetos de aprendizado de máquina. Oferece trilha de experimentos, projetos, registro de modelos e ferramentas de implantação. Agnóstica em relação à linguagem e ambiente de execução, facilita o desenvolvimento, reprodução e implantação de soluções de ML.

## Gradio 

Gradio é uma ferramenta de código aberto para criar interfaces de usuário simples para modelos de aprendizado de máquina. Facilita a criação de aplicativos interativos para visualizar e interagir com modelos ML sem a necessidade de conhecimento em programação de interface.

## Treino e Tracking:
Utilizando o mlflow foram realizados 4 experimentos, a seguir serão expostas as métricas dos dois melhores experimentos capturados com o MLflow(o modelo é avaliado por steps):

| Métricas | Experimento 1 | Experimento 2 | 
|:-----------:|:-----------:|:-----------:|
|  Acurácia    |   0.9648862512363996    |   0.9653808110781404  |  
|   f1    |   	0.9673590504451038   |     0.9678217821782177  |  
|   loss    |   0.043955136090517044  |   0.04509175568819046  | 
|  roc_auc    |  0.9795994065281898   |  0.9795375865479723   | 
|  Épocas    |  2    |  1    |  

### Acurácia x Steps:

<table>
  
  <tr>
   Experimento 1 <img src="https://github.com/ClaudianoLeonardo/ASR-NewsClassifier-PTBR/blob/main/images/acursxsteps1.png" alt="Experimento 1" width="800">  
  Experimento 2 <img src="https://github.com/ClaudianoLeonardo/ASR-NewsClassifier-PTBR/blob/main/images/acursxsteps2.png" alt="Experimento 2" width="800">
  </tr>

</table>

### Loss x Steps:

<table>
  
  <tr>
   Experimento 1 <img src="https://github.com/ClaudianoLeonardo/ASR-NewsClassifier-PTBR/blob/main/images/loss1.png" alt="Experimento 1" width="800">  
  Experimento 2 <img src="https://github.com/ClaudianoLeonardo/ASR-NewsClassifier-PTBR/blob/main/images/loss2.png" alt="Experimento 2" width="800">
  </tr>

</table>

