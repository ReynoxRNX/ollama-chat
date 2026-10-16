# Chat Ollama

Une petite application de bureau pour discuter avec des modèles d'IA qui tournent **en local** grâce à [Ollama](https://ollama.com). Pas de compte, pas de cloud, et **aucune dépendance Python à installer** (Tkinter + bibliothèque standard).

*English: a tiny, dependency-free desktop chat UI (Tkinter) for local models served by Ollama. See the usage steps below, they are language-independent.*

## Fonctionnalités

- Réponses affichées au fil de l'eau (streaming), avec bouton **Arrêter**
- Choix du modèle parmi ceux installés dans Ollama, température réglable
- Consigne système optionnelle (ex. « Réponds toujours en français »)
- Affichage de la « réflexion » des modèles de raisonnement (Qwen3, DeepSeek R1…)
- Vitesse de génération (tokens/s) après chaque réponse
- Export de la conversation en Markdown
- Fonctionne sur Windows, macOS et Linux

## Prérequis

1. **[Ollama](https://ollama.com/download)** installé et lancé.
2. **Python 3.8 ou plus récent** avec Tkinter :
   - Windows / macOS : l'installeur de [python.org](https://www.python.org/downloads/) l'inclut.
   - Linux (Debian/Ubuntu) : `sudo apt install python3-tk`
3. Au moins un modèle téléchargé, par exemple :

   ```bash
   ollama pull qwen3:8b
   ```

   Choisissez un modèle adapté à votre mémoire : un fichier de modèle doit tenir dans votre RAM / VRAM avec de la marge.

## Lancer l'application

```bash
git clone https://github.com/ReynoxRNX/ollama-chat.git
cd ollama-chat
python ollama_chat.py        # ou python3 sous macOS / Linux
```

**Dans VS Code** : ouvrez le dossier (`Fichier > Ouvrir le dossier…`), ouvrez `ollama_chat.py`, puis cliquez sur le bouton ▶ en haut à droite (ou `F5`). L'extension Python de Microsoft est conseillée.

Utilisation : choisissez un modèle, écrivez votre message, `Entrée` pour envoyer, `Maj+Entrée` pour un saut de ligne, `Échap` pour arrêter une réponse.

## Configuration

Par défaut l'application contacte `http://localhost:11434`. Pour utiliser un autre serveur Ollama (autre machine du réseau, port différent), définissez la variable d'environnement `OLLAMA_HOST` avant de lancer :

```bash
OLLAMA_HOST=192.168.1.20:11434 python ollama_chat.py          # macOS / Linux
$env:OLLAMA_HOST="192.168.1.20:11434"; python ollama_chat.py   # PowerShell
```

## Dépannage

| Message | Cause probable |
| --- | --- |
| « Impossible de joindre Ollama… » | Ollama n'est pas lancé : ouvrez l'application Ollama ou tapez `ollama serve`, puis cliquez sur *Actualiser*. |
| « Aucun modèle installé » | Téléchargez-en un avec `ollama pull <nom>`. |
| `ModuleNotFoundError: tkinter` | Installez Tkinter (voir Prérequis). |
| Réponses très lentes | Modèle trop gros pour votre machine : essayez-en un plus petit. |

## Limites connues

- Le texte est affiché brut : pas de rendu Markdown (titres, gras, coloration du code).
- Texte uniquement : pas d'envoi d'images ni de fichiers.
- Ollama tronque l'historique au-delà de la taille de contexte du modèle ; pour les très longues conversations, démarrez-en une nouvelle.

## Structure du projet

```
ollama_chat.py      interface Tkinter
ollama_client.py    client HTTP pour l'API d'Ollama (testable sans interface)
tests/              tests unitaires (faux serveur Ollama, aucun modèle requis)
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m doctest ollama_client.py
```

## Contribuer

Les issues et pull requests sont les bienvenues. Merci de lancer les tests avant de proposer une modification.

## Licence

[MIT](LICENSE)
