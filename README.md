# # 👁️ Neural Search - Identificador Forense

> Uma aplicação desktop para identificação e análise de similaridade de imagens utilizando Deep Learning.

![Status](https://img.shields.io/badge/Status-Funcional-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![AI](https://img.shields.io/badge/AI-CLIP%20Model-orange)

## 📄 Sobre o Projeto

O **Neural Search** é uma ferramenta desenvolvida para auxiliar na identificação de padrões visuais. Diferente de uma comparação simples de pixels, este projeto utiliza o modelo **CLIP (Contrastive Language-Image Pre-training)** para entender o *conteúdo semântico* da imagem.

Isso significa que o sistema é capaz de identificar imagens semelhantes mesmo que tenham tamanhos, iluminações ou enquadramentos diferentes, sendo ideal para cenários de **análise forense digital**, organização de bancos de dados ou verificação de direitos autorais.

### ✨ Funcionalidades Principais
* **Comparação Semântica:** Utiliza *embeddings* vetoriais para calcular a similaridade entre imagens.
* **Banco de Dados Local:** Permite salvar e indexar novas imagens de referência dinamicamente.
* **Interface Moderna:** GUI responsiva com modo escuro (Dark Mode).
* **Feedback em Tempo Real:** Mostra a porcentagem de similaridade com as imagens do banco.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Interface Gráfica (GUI):** [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
* **Inteligência Artificial:** [Sentence-Transformers](https://www.sbert.net/) (Modelo `clip-ViT-B-32`)
* **Processamento de Imagem:** Pillow (PIL)
* **Matemática/Tensores:** PyTorch

---

## 🤝 Desenvolvimento e Colaboração

Este projeto foi desenvolvido como parte do meu portfólio de **Desenvolvimento Python e Integração de IA**.

* **Lógica e Backend:** Implementação da lógica de carregamento do modelo, manipulação de arquivos, cálculo de vetores (embeddings) e integração com o sistema operacional.
* **Interface Visual (UI/UX):** A estrutura visual e o design da interface (botões, layout, tema) foram desenvolvidos com o auxílio de ferramentas de IA generativa, focando em boas práticas de usabilidade e estética moderna com a biblioteca `customtkinter`.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python instalado e execute os passos abaixo:

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git](https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git)
   cd SEU-REPOSITORIO

