# This scripts implement an LLM-based semantic clustering method for intent topic clustering.
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import threading
from typing import Literal
from openai import OpenAI
import json

from pydantic import BaseModel, create_model
from tqdm import tqdm

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

def create_literal_type(values) -> type[Literal]: # type: ignore
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
    for intent in tqdm(intents):
        topic = call_llm_assign_topic(intent, topic_list)
        clusters[topic].append(intent)
    
    return clusters

def parallel_cluster_intents(sites2intents: dict[str, list[str]], 
                           max_workers: int = 3,
                           chunk_size: int = 100) -> dict[str, dict[str, list[str]]]:
    """
    Runs clustering in parallel for multiple websites using ThreadPoolExecutor.
    
    Args:
        sites2intents: dictionary mapping website strings to lists of intents
        max_workers: Maximum number of parallel threads to use
        chunk_size: Size of chunks for clustering
    
    Returns:
        dictionary mapping websites to their clustering results
    """
    # Create a thread-safe progress bar
    pbar_lock = threading.Lock()
    pbar = tqdm(total=len(sites2intents), desc="Processing websites")
    
    def cluster_website(website: str, intents: list[str]) -> tuple[str, dict]:
        """
        Clusters intents for a single website and updates progress bar.
        Returns tuple of (website, clustering_result).
        """
        try:
            # Extract website name and format it
            website_name = eval(website)[0]
            formatted_name = website_name.replace("//", "_").replace("/", "_")
            
            # Perform clustering
            result = cluster_intents(intents, chunk_size=chunk_size)
            
            # Update progress bar thread-safely
            with pbar_lock:
                pbar.update(1)
                pbar.set_description(f"Completed {website_name}")
            
            return formatted_name, result
            
        except Exception as e:
            print(f"Error processing {website}: {str(e)}")
            return website, {}

    # Store all results
    all_results = {}
    
    # Create thread pool and submit jobs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all clustering jobs
        future_to_website = {
            executor.submit(cluster_website, website, intents): website
            for website, intents in sites2intents.items()
        }
        
        # Process completed jobs
        for future in as_completed(future_to_website):
            website = future_to_website[future]
            try:
                website_name, result = future.result()
                if result:  # Only store if we got valid results
                    all_results[website_name] = result
            except Exception as e:
                print(f"Failed to process {website}: {str(e)}")
    
    pbar.close()
    return all_results

def save_results(results: dict[str, dict[str, list[str]]], output_dir: str = "./"):
    """
    Saves clustering results to individual JSON files.
    
    Args:
        results: dictionary mapping websites to their clustering results
        output_dir: Directory to save the JSON files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for website, clusters in results.items():
        output_file = output_path / f"clusters_{website}.json"
        with open(output_file, 'w') as f:
            json.dump(clusters, f, indent=2)
        print(f"Saved clustering results for {website} to {output_file}")

if __name__ == "__main__":
    # Load input data
    if Path("sites2intents.json").exists():
        with open("sites2intents.json") as f:
            sites2intents = json.load(f)
        
        # Run parallel clustering
        results = parallel_cluster_intents(
            sites2intents,
            max_workers=20,  # Adjust based on your API rate limits and needs
            chunk_size=100
        )
        
        # Save results
        save_results(results)
    else:
        print("sites2intents.json not found! The sites2intents.json file should contain a dictionary mapping websites to lists of intents.")
    
