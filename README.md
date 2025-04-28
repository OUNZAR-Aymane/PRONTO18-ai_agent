# Agent Conversationnel RAG avec LangChain, Mistral AI, et Streamlit


## Présentation

Ce projet propose un **chatbot intelligent**, conçu pour répondre à des questions basées sur une base documentaire interne (PDFs techniques, manuels, documents de formation).  
Il utilise un pipeline **RAG** (Retrieval-Augmented Generation) combinant :

- **Mistral AI** (`mistral-large-latest` pour la génération de texte)
- **Mistral-Embed** pour les embeddings sémantiques
- **LangChain** pour l'orchestration de la recherche et du dialogue
- **FAISS** pour l'indexation vectorielle locale
- **Streamlit** pour l'interface web utilisateur

Le tout est **conteneurisé avec Docker** afin d'assurer portabilité, reproductibilité et facilité de déploiement.

---

## Fonctionnalités

- **Chat conversationnel contextuel** basé sur l'historique
- **Recherche intelligente** dans les documents PDF internes
- **Extraction d'images et OCR** pour traiter des documents scannés
- **Interface utilisateur web** Streamlit responsive
- **Utilisation professionnelle de VS Code avec Docker**

---

## Architecture Technique

| Composant | Description | Technologies utilisées |
|:----------|:------------|:------------------------|
| **Indexation des Documents** | Analyse de fichiers PDF, découpage en chunks, génération d'embeddings vectoriels | `embedder.py`, `embedder_with_OCR.py`, `pdf2image`, `pytesseract`, `Mistral-Embed`, `FAISS` |
| **Récupération augmentée** (RAG) | Reformulation de question + récupération de documents pertinents | `LangChain` (`create_history_aware_retriever`, `create_retrieval_chain`) |
| **Génération de réponse** | Construction d'une réponse basée sur les documents récupérés | `ChatMistralAI` (`mistral-large-latest`) |
| **Interface utilisateur** | Application web interactive | `Streamlit` (`interface.py`) |
| **Orchestration** | Scripts et notebooks de gestion | `chatBot.py`, `multimodal_embedder.ipynb` |

---

## Pipeline RAG utilisé

1. **Indexation**
    - Les fichiers PDF sont analysés, découpés en segments (`chunk_size=1000`, `overlap=200`).
    - Chaque segment est encodé en vecteur via **Mistral-Embed**.
    - Les vecteurs sont stockés dans **FAISS** (`faiss_index/`).

2. **Retrieval**
    - Lorsqu'une question est posée, elle est reformulée de manière contextuelle par **LangChain**.
    - Les chunks pertinents sont récupérés via une recherche FAISS.

3. **Génération**
    - Les passages retrouvés sont injectés dans un prompt structuré.
    - Le modèle **Mistral Large** génère une réponse complète en français.

---

## Technologies

- **LangChain**
- **MistralAI (chat et embeddings)**
- **FAISS**
- **pdf2image**, **pytesseract**, **PyMuPDF**
- **Streamlit**
- **Docker / Docker Compose**
- **VS Code Remote Containers**

---

## Installation & Utilisation

### 1. Prérequis

- Docker Desktop 24+
- Git 2.40+
- VS Code 1.85+ (extensions recommandées : *Dev Containers*, *Python*, *Jupyter*)
- Compte Mistral AI et clé API valide

### 2. Cloner le dépôt

```bash
git clone https://gitlab-df.imt-atlantique.fr/a24ounza/ai-agent.git
cd ai-agent
```

### 3. Configurer les variables d'environnement
Skip

### 4. Construction des images Docker

```bash
docker compose build
```

### 5. Lancer les conteneurs

- **Shell de développement** :
```bash
docker compose up dev
```
- **JupyterLab** :
```bash
docker compose up jupyter
```
Puis ouvrir [http://localhost:8888](http://localhost:8888)

- **Streamlit** :
```bash
docker compose up streamlit
```
Puis ouvrir [http://localhost:8501](http://localhost:8501)

### 6. Indexation des documents

Skip

### 7. Utiliser VS Code avec les conteneurs

- Ouvrir `ai-agent/` dans VS Code.
- `F1` > **Dev Containers: Reopen in Container**.
- Travailler directement dans l'environnement conteneurisé.

---

## Arborescence du projet

```bash
ai-agent/
├── Dockerfile
├── docker-compose.yml
├── .env
├── docs/
├── faiss_index/
├── workspace/
│   ├── chatBot.py
│   ├── embedder.py
│   ├── embedder_with_OCR.py
│   ├── multimodal_embedder.ipynb
|   ├── interface.py
```
