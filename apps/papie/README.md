# PAPIE (Personal AI Personal Assistant)

Streaming personal assistant chatbot with memory and task management.

## Overview

PAPIE is a streaming personal assistant chatbot that provides conversational AI with memory retention and task assistance capabilities.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PAPIE                                  │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py chat "What is machine learning?"        │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import PAPIE                        │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/chat_history.csv                      │
│  Model: LLM Chat (Gemini Streaming)                         │
│  Features: conversation_context, memory, tasks             │
│  Output: Contextual responses with streaming               │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer requests
```

## Usage

### CLI

```bash
# Chat with assistant
python cli.py chat "What is machine learning?"

# Continue conversation
python cli.py chat --history chat_history.json "Tell me more"

# Get assistant status
python cli.py status
```

### Streamlit UI

```bash
streamlit run apps/papie/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/papie/notebooks/chatbot_interaction.ipynb
```

## Project Structure

```
papie/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── chat_history.csv
└── notebooks/
    └── chatbot_interaction.ipynb
```

## License

See parent directory for license information.
