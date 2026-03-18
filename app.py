import os
import gradio as gr
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 1. CONFIGURACIÓN DEL MODELO GRATUITO
# Recuerda añadir tu HUGGINGFACEHUB_API_TOKEN en Settings -> Secrets del Space
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)
model = ChatHuggingFace(llm=llm)

# 2. DEFINICIÓN DEL GRAFO (LangGraph)
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    # Invocamos al modelo con los mensajes actuales
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Construcción del grafo
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)
app = workflow.compile()

# 3. INTERFAZ DE GRADIO
def predict(message, history):
    # Ejecutamos el grafo
    # El historial de Gradio se puede ignorar o mapear, aquí lo hacemos simple:
    inputs = {"messages": [("user", message)]}
    result = app.invoke(inputs)
    # Retornamos el contenido del último mensaje
    return result["messages"][-1].content

demo = gr.ChatInterface(
    fn=predict, 
    title="Mi Chatbot LangGraph Gratis",
    description="Usando Llama 3 vía Hugging Face Inference API"
)

if __name__ == "__main__":
    demo.launch()