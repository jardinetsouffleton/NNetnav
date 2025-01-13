# This scripts implement an LLM-based semantic clustering method for intent topic clustering.
import math
from typing import Literal
from openai import OpenAI
import json

from pydantic import BaseModel, create_model

# Algorithm:
#     Input:
#         intents (list[str]): a list of intents
#         chunk_size (int): the number of intents in each chunk (default: len(intents))
#     Output:
#         clusters (dict[str, list[str]]): a list of topics, and each topic contains a list of intents
#     Guarantee:
#         set(clusters.values()) == set(intents)
#     Procedure:
#         chunks = chunkify(intents, max_size=chunk_size)
#         topics_list = []
#         for chunk in chunks:
#             topics = call_llm_find_topics(chunk)
#             topics_list.extend(topics)
#         topic_list = call_llm_unique_topics(topics_list)
#         clusters = {}
#         for topic in topic_list:
#             clusters[topic] = []
#         for intent in intents:
#             topic = call_llm_assign_topic(intent, topic_list)
#             clusters[topic].append(intent)
#         return clusters

class Topics(BaseModel):
    topics: list[str]

def create_literal_type(values):
    return Literal[tuple(values)]  # type: ignore

def create_dynamic_model(field_name: str, allowed_values: list[str]) -> type[BaseModel]:
    DynamicLiteral = create_literal_type(allowed_values)
    return create_model(
        'DynamicModel',
        **{field_name: (DynamicLiteral, ...)}  # ... means the field is required
    )


def cluster_intents(intents: list[str], chunk_size: int = None) -> dict[str, list[str]]:
    if chunk_size is None:
        chunk_size = len(intents)
    
    client = OpenAI()  # Assumes you have OPENAI_API_KEY in environment variables

    def chunkify(items: list[str], max_size: int) -> list[list[str]]:
        return [items[i:i + max_size] for i in range(0, len(items), max_size)]

    def call_llm_find_topics(chunk: list[str]) -> list[str]:
        
        prompt = f"""Given these intents:
{json.dumps(chunk, indent=2)}

Generate a list of high-level topics that these intents fall under. 
Return the response as a JSON array of strings.
Be specific but not too granular - aim for {int(math.sqrt(chunk_size))}-{chunk_size // 2} topics for this set of intents."""

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=Topics,
        )
        
        return response.choices[0].message.parsed.topics

    def call_llm_unique_topics(topics: list[str]) -> list[str]:
        prompt = f"""Given these topics:
{json.dumps(topics, indent=2)}

Consolidate these into a unique set of high-level topics, merging similar ones.
Return the response as a JSON array of strings.
Be specific but not too granular - aim for concise, clear topics."""

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=Topics,
        )
        
        return response.choices[0].message.parsed.topics

    def call_llm_assign_topic(intent: str, topics: list[str]) -> str:
        DynamicModel = create_dynamic_model('topic', topics)

        prompt = f"""Given this intent:
"{intent}"

And these possible topics:
{json.dumps(topics, indent=2)}

Choose the most appropriate topic for this intent.
Return the chosen topic string."""

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=DynamicModel,
        )
        
        return response.choices[0].message.parsed.topic

    # Main algorithm implementation
    chunks = chunkify(intents, chunk_size)
    topics_list = []
    
    # Find topics for each chunk
    for chunk in chunks:
        topics = call_llm_find_topics(chunk)
        topics_list.extend(topics)
    
    # Consolidate topics
    topic_list = call_llm_unique_topics(topics_list)
    
    # Initialize clusters
    clusters = {topic: [] for topic in topic_list}
    
    # Assign intents to topics
    for intent in intents:
        topic = call_llm_assign_topic(intent, topic_list)
        clusters[topic].append(intent)
    
    return clusters

# Example usage:
if __name__ == "__main__":
    example_intents = [
        "book a flight",
        "check flight status",
        "cancel reservation",
        "find restaurants",
        "make dinner reservation",
        "view menu",
        "check weather",
        "get forecast"
    ]
    
    result = cluster_intents(example_intents, chunk_size=4)
    print(json.dumps(result, indent=2))
    
