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
#### Créer une clé API de Mistral 
1. Accéder à [https://console.mistral.ai/home](https://console.mistral.ai/home). Créer un compte ou ce connecter avec un compte google.

![SideBar](assets\images\sidebar.png)

2. Accéder à `Clés API`. *
3. Il faudra choisir un plan pour activer un forfait. Choisissez le plan Experiment (Gratuit ).
4. Accépter les conditions et vérifier votre numéro de téléphone.
![alt text](assets\images\api_key.png)
5. Clicker sur `Créer une nouvelle clé`. Choisir un nom pour la clé et une date d'expiration (Jamais).
6. Copier la clé et garder la.
#### Executer la commande 
```bash
echo MISTRAL_API_KEY="Votre clé API ici" > .env
```
### 4. Configuration de `config.yaml` pour l'authentification

Cette application utilise [`streamlit-authenticator`](https://github.com/mkhorasani/streamlit-authenticator) pour gérer la connexion et les sessions utilisateurs.  
Pour sécuriser vos identifiants et les cookies de session, suivez les étapes ci-dessous pour créer votre propre `config.yaml`.

### 1. Ne pas utiliser directement `config.example.yaml`

Ce fichier est un modèle. Vous devez créer votre **propre version sécurisée** et ne jamais la publier dans Git.

#### 2. Créez un fichier `config.yaml`

Créez un fichier `config.yaml` à la racine du projet à partir du modèle :

```bash
cp config.example.yaml config.yaml
```

Modifiez les valeurs à l’intérieur, en particulier :

##### Générer une `cookie.key` sécurisée

Cette clé permet de signer les cookies de session afin d’empêcher toute falsification. Elle doit :

- Contenir au moins 16 caractères aléatoires
- Rester secrète (ne jamais la publier)
- Être unique pour chaque instance

Pour en générer une avec Python :

```python
import secrets
print(secrets.token_urlsafe(32))
```

Collez la chaîne générée dans votre `config.yaml` :

```yaml
cookie:
  name: streamlit_auth
  key: "clé_sécurisée_générée_ici"
  expiry_days: 7
```
##### Étapes :

1. Ouvrir le fichier :

```bash
nano config.yaml
```

2. Naviguer avec les flèches et modifier :

```yaml
key: "nouvelle_clé_cookie_secure"
```

3. Enregistrer : `Ctrl + O` puis `Entrée`  
4. Quitter : `Ctrl + X`

⚠️ Attention à l’indentation : utilisez **des espaces, pas de tabulations**, et gardez les niveaux de retrait YAML intacts.
#### 4. Ajoutez `config.yaml` au `.gitignore`

Dans votre fichier `.gitignore`, ajoutez :

```
config.yaml
```

Cela empêchera toute fuite accidentelle d’informations sensibles.

#### Résumé

| Fichier                | Doit être publié ? | Remarques                           |
|------------------------|--------------------|-------------------------------------|
| `config.example.yaml` | ✅ Oui             | Fichier modèle, sans données réelles |
| `config.yaml`          | ❌ Non             | Contient des secrets — **ne pas publier** |

### 4. Construction des images Docker

```bash
docker compose build
```

### 5. Lancer les conteneurs

- **Shell de développement (Obligatoire pour accéder et mofifier le code)** :
```bash
docker compose up dev
```
- **JupyterLab (Si vous voulez utiliser ou ajouter des notebooks Jupyter)** :
```bash
docker compose up jupyter
```
Puis ouvrir [http://localhost:8888](http://localhost:8888)

- **Streamlit (Pour lancer le Chatbot)** :
```bash
docker compose up streamlit
```
Puis ouvrir [http://localhost:8501](http://localhost:8501)


### 7. Utiliser VS Code avec les conteneurs

- Ouvrir `ai-agent/` dans VS Code.
- `F1` > **Dev Containers: Reopen in Container**.
- Travailler directement dans l'environnement conteneurisé.
