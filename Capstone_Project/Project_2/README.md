# 🤖 Banking Customer Support AI

An intelligent customer support system that uses AI-powered routing and feedback learning to handle customer inquiries in banking environments.

## 📋 Overview

This application is a Streamlit-based Banking Customer Support AI that intelligently classifies and routes customer messages using OpenAI's GPT-4o-mini model. The system handles three types of interactions:

- **Positive Feedback** – Customer compliments and satisfaction messages
- **Negative Feedback** – Customer complaints and issues
- **Ticket Queries** – Customer requests for ticket status or assistance

The system learns from user feedback through few-shot learning, continuously improving response quality over time.

## ✨ Features

### 1. **Chat Interface**
- Submit customer messages in real-time
- Automatic sentiment and intent classification
- Agent-based routing to appropriate handlers
- Real-time database lookups for ticket status
- User feedback collection for continuous improvement

### 2. **Model Evaluation**
- **Response Quality Testing** – Evaluate accuracy, empathy, and clarity of AI responses
- **Routing Accuracy Testing** – Run comprehensive QA test cases to verify classification logic
- Performance metrics and success rate tracking

### 3. **History & Logs**
- Complete interaction history with timestamps
- Agent performance metrics (total interactions, success rate, tickets processed)
- Feedback integration tracking
- Ticket activity log
- Interactive filtering and search
- CSV export for analysis
- Analytics dashboard with charts and trends

### 4. **Intelligent Learning**
- Few-shot learning from successful interactions
- Feedback-driven improvement loop
- Tracks accuracy, empathy, and clarity scores
- Dynamic prompt enhancement with successful examples

## 🛠️ Tech Stack

- **Streamlit** – Interactive web UI
- **LangGraph** – Workflow orchestration and state management
- **OpenAI API** – GPT-4o-mini for NLP and classification
- **Pydantic** – Data validation
- **Pandas** – Data manipulation and analysis
- **Python-dotenv** – Environment variable management

## 🚀 Setup Instructions

### Prerequisites
- Python 3.13 or higher
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. **Clone/Navigate to the project:**
   ```bash
   cd Project_2
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate     # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit langgraph openai pydantic python-dotenv pandas
   ```

5. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

The app will be available at `http://localhost:8501`

## 📊 Project Structure

```
Project_2/
├── streamlit_app.py          # Streamlit UI application (all UI/presentation code)
├── main_agents.py            # Business logic (agent functions, workflow orchestration)
├── tickets.csv               # Ticket database for status lookups
├── .env                      # Environment variables (API keys)
├── .venv/                    # Virtual environment
├── Project_2.ipynb           # Jupyter notebook (analysis/experimentation)
├── README.md                 # This file
└── Screenshots/              # Documentation screenshots
```

### File Responsibilities
- **streamlit_app.py** – Contains all Streamlit UI code, page navigation, form handling, metrics display, and Streamlit context operations
- **main_agents.py** – Pure business logic: agent routing, OpenAI API calls, few-shot learning, workflow orchestration with LangGraph

## 🏗️ Architecture

```
streamlit run streamlit_app.py
    ↓
┌─────────────────────────────────────────────┐
│        streamlit_app.py (UI Layer)          │
│  • Chat Interface page                      │
│  • Model Evaluation page                    │
│  • History & Logs page                      │
│  • Session state management                 │
│  • Streamlit components & forms             │
└────────────┬────────────────────────────────┘
             │ imports
             ↓
┌─────────────────────────────────────────────┐
│     main_agents.py (Business Logic)         │
│  • Agent routing & classification           │
│  • OpenAI API integration                   │
│  • Few-shot learning                        │
│  • LangGraph workflow orchestration         │
│  • Pure functions (no Streamlit)            │
└─────────────────────────────────────────────┘
             │ reads/writes
             ↓
┌─────────────────────────────────────────────┐
│  External Resources                         │
│  • OpenAI API (GPT-4o-mini)                │
│  • tickets.csv (ticket database)            │
│  • .env (API keys)                          │
└─────────────────────────────────────────────┘
```
**Last Updated:** May 2026  
**Status:** ✅ Active and Running
