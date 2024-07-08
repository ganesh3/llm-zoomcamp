from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.properties import StringProperty, BooleanProperty
from kivy.clock import Clock
from openai import OpenAI
from elasticsearch import Elasticsearch

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
es_client = Elasticsearch('http://localhost:9200/')

def elastic_search(query, index_name = "course-questions-docker"):
    """
    Searches the Elasticsearch index for the given query and returns the top 5 matching documents.

    Args:
    query (str): The search query to be used.
    index_name (str, optional): The name of the Elasticsearch index to search in. Defaults to "course-questions-docker".

    Returns:
    list: A list of dictionaries containing the top 5 matching documents from the Elasticsearch index.

    Raises:
    Exception: If there is an error in the API request or response.
    """
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    
    response_es = es_client.search(index=index_name, body=search_query)
    
    result_docs = []

    for hit in response_es['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

def build_prompt(query, search_results):
    """
    Builds a prompt for the LLM based on the given query and search results.

    Args:
    query (str): The question to be answered.
    search_results (list): A list of dictionaries containing the context for the question. Each dictionary should have the keys 'section', 'question', and 'text'.

    Returns:
    str: A formatted prompt string for the LLM.

    Raises:
    ValueError: If the search_results list is empty.
    """
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    """
    Uses the OpenAI API to generate a response to the given prompt.

    Args:
    prompt (str): The question or statement to be answered or responded to.

    Returns:
    str: The generated response from the OpenAI API.

    Raises:
    Exception: If there is an error in the API request or response.
    """
    response = client.chat.completions.create(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query):
    """
    This function identifies related content using Elasticsearch, builds a prompt for the LLM, calls the LLM to generate a response, and returns the answer.

    Args:
    query (str): The question to be answered.

    Returns:
    str: The generated response from the LLM.

    Raises:
    Exception: If there is an error in the API request or response.
    """
    print("--------------IDENTIFYING RELATED CONTENT USING ELASTIC SEARCH-------------------")
    search_results = elastic_search(query)
    print("--------------BUILDING PROMPT-------------------")
    prompt =  build_prompt(query, search_results)
    print("--------------CALLING THE LLM-------------------")
    answer = llm(prompt)
    print("--------------RETURNING THE ANSWER-------------------")
    return answer

class MyTextInputApp(App):
  input_text = StringProperty("")
  output_text = StringProperty("Initial Output")
  is_loading = BooleanProperty(False)
  window_size = (800, 600)

  def build(self):
    layout = BoxLayout(orientation="vertical")

    # Input box
    input_box = TextInput(hint_text="Enter text here", on_text_change=self.on_text_changed)

    # Ask button
    ask_button = Button(text="Ask", on_press=self.on_ask_clicked)

    # Output label
    output_label = Label(text=self.output_text)

    # Loading indicator
    loading_indicator = ActivityIndicator(active=self.is_loading)

    # Add elements to layout
    layout.add_widget(input_box)
    layout.add_widget(ask_button)
    layout.add_widget(output_label)
    layout.add_widget(loading_indicator)

    return layout

  def on_text_changed(self, text):
    self.input_text = text

  def on_ask_clicked(self, instance):
    self.is_loading = True
    self.output_text = "Processing..."  # Placeholder while processing

    # Simulate some processing time (replace with your actual rag function)
    Clock.schedule_once(lambda dt: self.finish_processing(), 2)

  def finish_processing(self):
    self.output_text = f"Output: {self.rag(self.input_text)}"  # Replace with your rag function call
    self.is_loading = False

if __name__ == "__main__":
  MyTextInputApp().run()