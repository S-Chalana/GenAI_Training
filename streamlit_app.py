import streamlit as st
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

# Step 1: Setup the Client
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(override=True, dotenv_path=env_path)
my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=my_api_key)

# Step 2: Define state structure to track input, decision, and output
class State(TypedDict):
    user_input: str
    decision: str
    output: str
    evaluation: dict  # Will store accuracy, empathy, clarity scores

class Route(BaseModel):
    step: Literal["positive", 'negative', 'query'] = Field(None, description="The next step in the routing process")

# Step 3: Function to determine the sentiment using AI
def get_router_response(user_input: str) -> str:
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

def get_few_shot_examples(agent_type: str, limit: int = 3):
    """Get successful examples from history for few-shot learning."""
    if 'history' not in st.session_state or not st.session_state['history']:
        return []
    
    df = pd.DataFrame(st.session_state['history'])
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

# Step 4: Define functions for each classification with feedback integration
def handle_positive(state: State) -> State:
    # Get successful examples for few-shot learning
    examples = get_few_shot_examples("positive")
    
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



def handle_negative(state: State) -> State:
    # Get successful examples for few-shot learning
    examples = get_few_shot_examples("negative")
    
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



def handle_query(state: State) -> State:
    # Extract ticket number from user input
    user_message = state['user_input']
    ticket_info = get_ticket_status(user_message)
    
    # Get successful examples for few-shot learning
    examples = get_few_shot_examples("query")
    
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

# Evaluation Function: Assess response quality using OpenAI
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

def evaluate_routing_accuracy() -> dict:
    """Test classification logic and calculate routing success rate."""
    results = [{"input": tc["input"], "expected": tc["expected_decision"], 
                "actual": get_router_response(tc["input"]), 
                "correct": get_router_response(tc["input"]) == tc["expected_decision"]} 
               for tc in QA_TEST_CASES]
    correct = sum(1 for r in results if r["correct"])
    return {"success_rate": (correct / len(QA_TEST_CASES)) * 100, "correct": correct, 
            "total": len(QA_TEST_CASES), "results": results}


# Step 5: Routing function to determine which classification function to call
def route_request(state: State) -> State:
    decision = get_router_response(state["user_input"])
    return {"decision": decision}

def route_decision(state: State) -> str:
    # Return the decision value, not the node name
    # The conditional edges mapping will translate it to the node name
    decision = state["decision"]
    if decision in ["positive", "negative", "query"]:
        return decision
    return "query"

# Step 6: Build LangGraph workflow
def build_workflow() -> StateGraph:
    workflow = StateGraph(State)
    workflow.add_node("handle_positive", handle_positive)
    workflow.add_node("handle_negative", handle_negative)
    workflow.add_node("handle_query", handle_query)
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

# Test Scenarios for Each Agent Role
TEST_SCENARIOS = {
    "positive": ["I love this service!", "Great job, thank you!", "Excellent support!"],
    "negative": ["This is terrible", "Very disappointed", "Poor service"],
    "query": ["Check ticket T001", "Status of T005?", "What about ticket T003?"]
}

def save_to_history(state: State, trace_data=None, feedback=None):
    """Save interaction to session history with traces and feedback."""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    ticket_id = re.search(r'T\d+', state['user_input'])
    entry = {
        "timestamp": pd.Timestamp.now(),
        "input": state['user_input'],
        "decision": state['decision'],
        "output": state['output'],
        "ticket_id": ticket_id.group(0) if ticket_id else None,
        "trace": trace_data,
        "feedback": feedback,
        "feedback_comments": None,  # Will be updated when user provides comments
        "success": None  # Will be set by user feedback
    }
    st.session_state['history'].append(entry)

def get_agent_metrics():
    """Calculate agent success/failure rates from feedback."""
    if 'history' not in st.session_state or not st.session_state['history']:
        return {}
    
    df = pd.DataFrame(st.session_state['history'])
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

# Step 7: Interactive Dashboard
def run_streamlit_app():
    st.title("🤖 Banking Customer Support AI")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to:", ["Chat Interface", "Model Evaluation", "History & Logs"])
    
    # Page 1: Chat Interface
    if page == "Chat Interface":
        st.subheader("💬 User Input & Agent Routing")
        user_input = st.text_area("Enter your message:")
        
        if st.button("Submit", type="primary"):
            if user_input:
                initial_state: State = {"user_input": user_input, "decision": "", "output": "", "evaluation": {}}
                workflow = build_workflow()
                final_state = workflow.invoke(initial_state)
                
                # Create trace data
                ticket_id = re.search(r'T\d+', user_input)
                trace_data = {
                    "classification_prompt": f"Classify: '{user_input}'",
                    "classification_output": final_state['decision'],
                    "agent_handler": f"handle_{final_state['decision']}",
                    "ticket_action": f"Queried tickets.csv for {ticket_id.group(0)}" if ticket_id else "No database query"
                }
                
                # Save to history with trace
                save_to_history(final_state, trace_data=trace_data)
                st.session_state['last_interaction'] = final_state
                st.session_state['last_trace'] = trace_data

                # Display response
                st.success(f"**Agent Response:** {final_state['output']}")
                
                # Display classification and routing
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<small>**Agent Classification:** {final_state['decision'].upper()}</small>", unsafe_allow_html=True)
                with col2:
                    ticket = re.search(r'T\d+', user_input)
                    st.markdown(f"<small>**Database Query:** {'Yes' if ticket else 'No'}</small>", unsafe_allow_html=True)
                
                
                # User Feedback for Improvement Loop
                st.subheader("📝 Provide Feedback")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful"):
                        st.session_state['history'][-1]['feedback'] = 'positive'
                        st.session_state['history'][-1]['success'] = True
                        st.success("Thanks for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.session_state['history'][-1]['feedback'] = 'negative'
                        st.session_state['history'][-1]['success'] = False
                        st.warning("Feedback recorded for improvement")
                
                # Additional feedback comments
                user_feedback_text = st.text_area("Additional comments (optional):", placeholder="Tell us what you liked or what could be improved...")
                if st.button("Submit Comments") and user_feedback_text:
                    st.session_state['history'][-1]['feedback_comments'] = user_feedback_text
                    st.success("Thank you for your detailed feedback!")
            else:
                st.warning("Please enter a message")
    
    # Page 2: Test Scenarios
    elif page == "Test Scenarios":
        st.subheader("🧪 Test Agent Scenarios")
        agent_type = st.selectbox("Select Agent Type:", ["positive", "negative", "query"])
        
        st.write(f"**Pre-built test cases for {agent_type} agent:**")
        for scenario in TEST_SCENARIOS[agent_type]:
            if st.button(f"Test: {scenario}", key=scenario):
                initial_state: State = {"user_input": scenario, "decision": "", "output": "", "evaluation": {}}
                workflow = build_workflow()
                final_state = workflow.invoke(initial_state)
                save_to_history(final_state)
                
                col1, col2 = st.columns(2)
                col1.metric("Expected", agent_type)
                col2.metric("Actual", final_state['decision'])
                st.write(f"**Response:** {final_state['output']}")
                if final_state['decision'] == agent_type:
                    st.success("✅ Routing Correct")
                else:
                    st.error("❌ Routing Failed")
    
    # Page 3: History & Logs
    elif page == "History & Logs":
        if 'history' in st.session_state and st.session_state['history']:
            df = pd.DataFrame(st.session_state['history'])
            
            # Agent Success Metrics
            metrics = get_agent_metrics()
            st.subheader("📈 Agent Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Interactions", metrics['total_interactions'])
            col2.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
            col3.metric("Tickets Processed", metrics['tickets_processed'])
            col4.metric("Unique Tickets", metrics['unique_tickets'])
            
            # Feedback Integration & Improvement Loop
            st.subheader("🔄 Agent Improvement Loop")
            st.info("**How it works:** The system learns from your feedback. Successful interactions (👍) are used as examples to improve future responses through few-shot learning.")
            
            feedback_df = df[df['feedback'].notna()]
            if len(feedback_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Feedback by Agent Type:**")
                    agent_feedback = feedback_df.groupby(['decision', 'feedback']).size().unstack(fill_value=0)
                    st.dataframe(agent_feedback)
                
                with col2:
                    st.write("**Learning Examples Active:**")
                    for agent_type in ['positive', 'negative', 'query']:
                        examples = get_few_shot_examples(agent_type, limit=3)
                        st.write(f"**{agent_type.capitalize()}:** {len(examples)} example(s)")
                        if examples:
                            st.success(f"✅ Agent is learning from {len(examples)} successful interaction(s)")
                
                # Show improvement trend over time
                st.write("**Success Rate Trend:**")
                feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
                daily_success = feedback_df.groupby('date')['feedback'].apply(
                    lambda x: (x == 'positive').sum() / len(x) * 100 if len(x) > 0 else 0
                )
                if len(daily_success) > 0:
                    st.line_chart(daily_success)
            else:
                st.warning("No feedback yet. Start providing feedback to enable the improvement loop!")
            
            # Ticket Activity Log
            if metrics['tickets_processed'] > 0:
                st.subheader("🎫 Ticket Activity Log")
                ticket_df = df[df['ticket_id'].notna()][['timestamp', 'ticket_id', 'decision', 'feedback']]
                st.dataframe(ticket_df, use_container_width=True)
            
            # Filters
            st.subheader("🔍 Interaction Logs")
            col1, col2 = st.columns(2)
            with col1:
                filter_decision = st.multiselect("Filter by Agent:", ["positive", "negative", "query"], default=["positive", "negative", "query"])
            with col2:
                if st.button("Clear History"):
                    st.session_state['history'] = []
                    st.rerun()
            
            # Display filtered logs
            filtered_df = df[df['decision'].isin(filter_decision)]
            st.dataframe(filtered_df[['timestamp', 'input', 'decision', 'ticket_id', 'feedback']], use_container_width=True)
            
            # Prompt Traces, Classification Output, and Ticket Actions
            st.subheader("🔍 Prompt Traces & Classification Details")
            st.write("Select an interaction to view detailed traces:")
            if len(filtered_df) > 0:
                selected_idx = st.selectbox("Choose interaction:", 
                                           range(len(filtered_df)), 
                                           format_func=lambda i: f"{filtered_df.iloc[i]['timestamp']} - {filtered_df.iloc[i]['input'][:50]}...")
                if selected_idx is not None:
                    selected_row = filtered_df.iloc[selected_idx]
                    
                    # Show trace data
                    if selected_row.get('trace'):
                        st.json(selected_row['trace'])
                    else:
                        st.info("No trace data available for this interaction")
                    
                    # Show user feedback comments if available
                    if selected_row.get('feedback_comments'):
                        st.subheader("💬 User Feedback Comments")
                        st.write(selected_row['feedback_comments'])
            
            # Download logs
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download Logs", csv, "interaction_logs.csv", "text/csv")
            
            # Analytics
            st.subheader("📊 Analytics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Agent Distribution**")
                st.bar_chart(df['decision'].value_counts())
            with col2:
                st.write("**Recent Activity**")
                st.line_chart(df.groupby(df['timestamp'].dt.floor('min')).size())
        else:
            st.info("No interaction history yet. Start chatting to see logs!")
    
    # Page 4: Model Evaluation  
    elif page == "Model Evaluation":
        st.subheader("📊 Model Evaluation")
        if 'last_interaction' not in st.session_state:
            st.warning("Submit a query first to enable evaluation")
            return
            
        eval_tab1, eval_tab2 = st.tabs(["Response Quality", "Routing Accuracy"])
        
        with eval_tab1:
            if st.button("Evaluate Response"):
                with st.spinner("Evaluating..."):
                    eval_data = evaluate_response(st.session_state['last_interaction'])['evaluation']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{eval_data.get('accuracy', 0)}/5")
                    col2.metric("Empathy", f"{eval_data.get('empathy', 0)}/5")
                    col3.metric("Clarity", f"{eval_data.get('clarity', 0)}/5")
                    if eval_data.get('feedback'):
                        st.info(f"**Feedback:** {eval_data['feedback']}")
        
        with eval_tab2:
            if st.button("Test Routing"):
                with st.spinner("Testing..."):
                    results = evaluate_routing_accuracy()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Success Rate", f"{results['success_rate']:.1f}%")
                    col2.metric("Correct", results['correct'])
                    col3.metric("Total", results['total'])
                    df = pd.DataFrame(results['results'])
                    st.dataframe(df.style.apply(lambda row: ['background-color: #d4edda'] * len(row) if row['correct'] 
                                                else ['background-color: #f8d7da'] * len(row), axis=1), use_container_width=True)
                    st.bar_chart(df['actual'].value_counts())

if __name__ == "__main__":
    run_streamlit_app()

