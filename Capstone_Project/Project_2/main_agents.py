"""
Banking Customer Support AI - Agent Business Logic
Contains routing, handler functions, evaluation, and workflow orchestration.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import Literal
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import re
import json
from functools import partial

# Setup the Client
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(override=True, dotenv_path=env_path)
my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=my_api_key)

# Define state structure to track input, decision, and output
class State(TypedDict):
    user_input: str
    decision: str
    output: str
    evaluation: dict  # Will store accuracy, empathy, clarity scores

class Route(BaseModel):
    step: Literal["positive", 'negative', 'query'] = Field(None, description="The next step in the routing process")

# QA Test Cases for Classification Logic
QA_TEST_CASES = [
    {"input": "I love this service! Thank you so much!", "expected_decision": "positive"},
    {"input": "This is absolutely wonderful!", "expected_decision": "positive"},
    {"input": "I'm very disappointed with the service", "expected_decision": "negative"},
    {"input": "This is terrible and frustrating", "expected_decision": "negative"},
    {"input": "What's the status of ticket T001?", "expected_decision": "query"},
    {"input": "Can you check my order T005?", "expected_decision": "query"},
    {"input": "How do I track my ticket T003?", "expected_decision": "query"},
    {"input": "Great experience overall!", "expected_decision": "positive"},
    {"input": "Very poor quality", "expected_decision": "negative"},
    {"input": "Please help with T010", "expected_decision": "query"}
]

# Test Scenarios for Each Agent Role
TEST_SCENARIOS = {
    "positive": ["I love this service!", "Great job, thank you!", "Excellent support!"],
    "negative": ["This is terrible", "Very disappointed", "Poor service"],
    "query": ["Check ticket T001", "Status of T005?", "What about ticket T003?"]
}

# Routing and Classification Functions
def get_router_response(user_input: str) -> str:
    """Determine the sentiment/intent using OpenAI API."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Classify the sentiment as 'positive', 'negative', or 'query'. Respond with only one word: positive, negative, or query."
            },
            {
                "role": "user",
                "content": f"Classify the sentiment of the following input: '{user_input}'"
            }
        ],
        temperature=0.7
    )
    classify = response.choices[0].message.content.strip().lower()
    # Ensure we return a valid value
    if classify in ["positive", "negative", "query"]:
        return classify
    elif "positive" in classify:
        return "positive"
    elif "negative" in classify:
        return "negative"
    else:
        return "query"

def route_request(state: State) -> State:
    """Route the request based on classification."""
    decision = get_router_response(state["user_input"])
    return {"decision": decision}

def route_decision(state: State) -> str:
    """Determine which handler to use based on decision."""
    decision = state["decision"]
    if decision in ["positive", "negative", "query"]:
        return decision
    return "query"

# Handler Functions
def get_few_shot_examples(agent_type: str, limit: int = 3, history=None):
    """Get successful examples from history for few-shot learning.
    
    Args:
        agent_type: Type of agent (positive, negative, query)
        limit: Max number of examples to return
        history: List of historical interactions. If None, returns empty list.
    """
    if history is None or not history:
        return []
    
    df = pd.DataFrame(history)
    # Filter by agent type and positive feedback
    successful = df[(df['decision'] == agent_type) & (df['feedback'] == 'positive')]
    
    if len(successful) > 0:
        examples = []
        for _, row in successful.tail(limit).iterrows():
            examples.append({
                "input": row['input'],
                "output": row['output']
            })
        return examples
    return []

def handle_positive(state: State, history: list = None) -> State:
    """Handle positive feedback from customer."""
    # Get successful examples for few-shot learning
    examples = get_few_shot_examples("positive", history=history)
    
    system_prompt = "Thank the user for their positive input."
    if examples:
        system_prompt += "\n\nHere are examples of successful responses:\n"
        for ex in examples:
            system_prompt += f"\nUser: {ex['input']}\nAssistant: {ex['output']}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state['user_input']}
        ],
        max_tokens=500
    )
    return {"output": response.choices[0].message.content.strip()}

def handle_negative(state: State, history: list = None) -> State:
    """Handle negative feedback from customer."""
    # Get successful examples for few-shot learning
    examples = get_few_shot_examples("negative", history=history)
    
    system_prompt = "Apologise for the inconvenience."
    if examples:
        system_prompt += "\n\nHere are examples of successful responses:\n"
        for ex in examples:
            system_prompt += f"\nUser: {ex['input']}\nAssistant: {ex['output']}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state['user_input']}
        ],
        max_tokens=500
    )
    return {"output": response.choices[0].message.content.strip()}

def get_ticket_status(user_message: str) -> str:
    """Read ticket status from CSV file."""
    ticket_match = re.search(r'T\d+', user_message, re.IGNORECASE)
    if not ticket_match:
        return "No ticket number found"
    ticket_no = ticket_match.group(0).upper()
    csv_path = os.path.join(script_dir, "tickets.csv")
    try:
        df = pd.read_csv(csv_path)
        ticket_row = df[df['ticket_no'] == ticket_no]
        return f"Ticket {ticket_no}: {ticket_row.iloc[0]['status']}" if not ticket_row.empty else f"Ticket {ticket_no} not found"
    except:
        return "Unable to access ticket database"

def handle_query(state: State, history: list = None) -> State:
    """Handle ticket queries and information requests."""
    # Extract ticket number from user input
    user_message = state['user_input']
    ticket_info = get_ticket_status(user_message)
    
    # Get successful examples for few-shot learning
    examples = get_few_shot_examples("query", history=history)
    
    system_prompt = f"""Handle the user query. Ticket database lookup result: {ticket_info}
                If ticket found, respond as: "Your ticket #[TicketNumber] is currently marked as: [Status]."
                If not found, ask user to verify ticket number."""
    
    if examples:
        system_prompt += "\n\nHere are examples of successful responses:\n"
        for ex in examples:
            system_prompt += f"\nUser: {ex['input']}\nAssistant: {ex['output']}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=500
    )
    return {"output": response.choices[0].message.content.strip()}

# Evaluation Functions
def evaluate_response(state: State) -> State:
    """Evaluate the generated response for accuracy, empathy, and clarity."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""Evaluate this customer service response on a scale of 1-5:
User Query: {state['user_input']}
Agent Type: {state['decision']}
Response: {state['output']}

Rate: Accuracy (1-5), Empathy (1-5), Clarity (1-5)
Respond with JSON: {{"accuracy": X, "empathy": X, "clarity": X, "feedback": "brief explanation"}}"""
            }],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return {"evaluation": json.loads(response.choices[0].message.content)}
    except Exception as e:
        return {"evaluation": {"accuracy": 0, "empathy": 0, "clarity": 0, "feedback": f"Error: {str(e)}"}}

def evaluate_routing_accuracy() -> dict:
    """Test classification logic and calculate routing success rate."""
    results = [{"input": tc["input"], "expected": tc["expected_decision"], 
                "actual": get_router_response(tc["input"]), 
                "correct": get_router_response(tc["input"]) == tc["expected_decision"]} 
               for tc in QA_TEST_CASES]
    correct = sum(1 for r in results if r["correct"])
    return {"success_rate": (correct / len(QA_TEST_CASES)) * 100, "correct": correct, 
            "total": len(QA_TEST_CASES), "results": results}

# Workflow Building
def build_workflow(history: list = None) -> StateGraph:
    """Build the LangGraph workflow for agent routing and handling.
    
    Args:
        history: List of historical interactions for few-shot learning
    """
    workflow = StateGraph(State)
    
    # Use partial to bind history to handlers
    handle_positive_with_history = partial(handle_positive, history=history)
    handle_negative_with_history = partial(handle_negative, history=history)
    handle_query_with_history = partial(handle_query, history=history)
    
    workflow.add_node("handle_positive", handle_positive_with_history)
    workflow.add_node("handle_negative", handle_negative_with_history)
    workflow.add_node("handle_query", handle_query_with_history)
    workflow.add_node("route_request", route_request)

    workflow.add_edge(START, "route_request")
    workflow.add_conditional_edges(
        "route_request",
        route_decision,
        {
            "positive": "handle_positive",
            "negative": "handle_negative",
            "query": "handle_query",
        },
    )
    workflow.add_edge("handle_positive", END)
    workflow.add_edge("handle_negative", END)
    workflow.add_edge("handle_query", END)

    return workflow.compile()

# History and Metrics Functions
def save_to_history(state: State, trace_data=None, feedback=None, session_state=None):
    """Save interaction to session history with traces and feedback.
    
    Args:
        state: The agent state
        trace_data: Execution trace information
        feedback: User feedback
        session_state: Streamlit session state dict (pass from UI layer)
    """
    if session_state is None:
        return
    
    if 'history' not in session_state:
        session_state['history'] = []
    
    ticket_id = re.search(r'T\d+', state['user_input'])
    entry = {
        "timestamp": pd.Timestamp.now(),
        "input": state['user_input'],
        "decision": state['decision'],
        "output": state['output'],
        "ticket_id": ticket_id.group(0) if ticket_id else None,
        "trace": trace_data,
        "feedback": feedback,
        "feedback_comments": None,
        "success": None
    }
    session_state['history'].append(entry)

def get_agent_metrics(session_state=None):
    """Calculate agent success/failure rates from feedback.
    
    Args:
        session_state: Streamlit session state dict (pass from UI layer)
    
    Returns:
        Dictionary with agent metrics
    """
    if session_state is None or 'history' not in session_state:
        return {}
    
    history = session_state.get('history', [])
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    feedback_df = df[df['feedback'].notna()]
    
    metrics = {
        "total_interactions": len(df),
        "feedback_count": len(feedback_df),
        "positive_feedback": len(feedback_df[feedback_df['feedback'] == 'positive']),
        "negative_feedback": len(feedback_df[feedback_df['feedback'] == 'negative']),
        "tickets_processed": df['ticket_id'].notna().sum(),
        "unique_tickets": df['ticket_id'].nunique()
    }
    
    if len(feedback_df) > 0:
        metrics['success_rate'] = (metrics['positive_feedback'] / len(feedback_df)) * 100
    else:
        metrics['success_rate'] = 0
    
    return metrics
