import json
from math import degrees
from typing import List
import pickle
import matplotlib.pyplot as plt
from matplotlib import colormaps
from transformers import BertTokenizer
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Extra
from graphRagSystemPrompt import systemPrompt
import networkx as nx
import igraph as ig
import leidenalg
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np



tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#client = genai.Client(api_key="")

# reading the file
file_obj = open("text_corpus.txt", "r")
data = file_obj.read()
def chunkData(content, overlap=100, chunk_size=600):

    tokens = tokenizer.tokenize(content)
    chunks = []
    i = 0
    while i < len(tokens) - chunk_size:
        chunk = tokens[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def tokenTostr(chunks):
    strChunks = []
    for chunk in chunkData(data):
        strChunks.append(" ".join(chunk))

    return strChunks

strict_config = ConfigDict(extra="ignore")

class Entity(BaseModel):
    model_config = strict_config
    id: str
    type: str
    desc: str

class Relation(BaseModel):
    """Represents a relationship between two entities."""
    model_config = strict_config
    source: str
    target: str
    type: str
    desc: str

class GraphData(BaseModel):
    """The main model for the entire JSON structure."""
    model_config = strict_config
    entities: List[Entity]
    relations: List[Relation]





def entityRelationExtraction(chunks, systemPrompt):
    strChunks = tokenTostr(chunks)
    print("chunks received")
    entities = []
    relations = []

    for i in range(len(strChunks)):
        print(f"processing chunk {i}")
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            config={
                "temperature":0.1,
                "system_instruction":systemPrompt,
                "response_mime_type": "application/json",
                "response_schema": GraphData,
            },
            contents=strChunks[i]
        )

        body = json.loads(response.text)

        entities.append(body["entities"])
        relations.append(body["relations"])

    return entities, relations

# chunks = chunkData(data)
# entities, relations= entityRelationExtraction(chunks,systemPrompt)
# print("entities extracted")
#
# print("writing data to file")
# # writing data to a file
# with open("entities.dat", "wb") as f:
#     pickle.dump(entities, f)
#
# with open("relations.dat", "wb") as f:
#     pickle.dump(relations, f)

print("reading data from file")
with open("entities.dat", "rb") as f:
    entitiesInChunks = pickle.load(f)

with open("relations.dat", "rb") as f:
    relationsInChunks = pickle.load(f)



G = nx.Graph()
entities = []
relations = []

for entityChunk in entitiesInChunks:
    for entity in entityChunk:
        entities.append(entity)

for relationChunk in relationsInChunks:
    for relation in relationChunk:
        relations.append(relation)


for entity in entities:
    nodeId = entity["id"]
    entity_type = entity["type"]
    desc = entity["desc"]

    if nodeId in G:
        G.nodes[nodeId]["frequency"] += 1
    else:
        G.add_node(entity["id"], type=entity_type, description=desc, frequency=1)


for relation in relations:
    source = relation["source"]
    target = relation["target"]
    relation_type = relation["type"]

    if G.has_edge(source, target):
        G[source][target]["weight"] += 1

    else:
        G.add_edge(source, target, type=relation_type, description=relation["desc"], weight=1)

def printEntitiesAndRelations(G):
    print("Entities with frequency:")
    for node, data in G.nodes(data=True):
        print(node, data)

    print("\nRelations with weights:")
    for u, v, data in G.edges(data=True):
        print(u, v, data.get('type',"no type"), data.get("weight", 0))


# printEntitiesAndRelations(G)

def showNxGraph(G):
    nx.draw(G)
    plt.figure(figsize=(22,16))

    pos = nx.spring_layout(G, seed=25, k=1, scale=200)


    nodeDegrees = [G.degree(x) for x in G]
    edgeWeights = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=nodeDegrees, cmap=plt.cm.GnBu)

    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color='red'
    )
    plt.show()

showNxGraph(G)
def runningLeiden(G):
    node_names = list(G.nodes())
    frequencies = [G.nodes[n].get("frequency", 0) for n in node_names]


    G_ig = ig.Graph.TupleList(
        [(s, t, d['type'],d["weight"]) for s, t, d in G.edges(data=True)],
        edge_attrs=['type', "weight"],
        directed=False)
    G_ig.vs["frequency"] = frequencies

    partition = leidenalg.find_partition(
        G_ig,
        leidenalg.ModularityVertexPartition,
        weights=G_ig.es["weight"])

    membership = partition.membership
    print(f"membership: {membership}")
    G_ig.vs["community"] = membership

    node_labels = G_ig.vs['name']
    communities = [[node_labels[idx] for idx in community] for community in partition]

    # Normalize community IDs
    unique_comms = sorted(set(membership))
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    membership_norm = [comm_map[x] for x in membership]


    # Build color list
    num_communities = max(membership_norm) + 1
    palette = plt.get_cmap("tab20")
    colors = [palette(i % num_communities) for i in membership_norm]

    layout = G_ig.layout("fr")
    visual_style = {}
    visual_style["vertex_label"] = G_ig.vs["name"]
    visual_style["vertex_color"] = colors
    visual_style['edge_width'] = [w for w in G_ig.es['weight']]
    visual_style["layout"] = layout
    visual_style["vertex_size"] = 20
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 100

    print("plotting communities in graph.png")
    ig.plot(G_ig, target="graph.png", **visual_style)

    return G_ig, partition



graph, partition = runningLeiden(G)

def generateSummaries(graph, partition):
    community_edges = defaultdict(set)
    community_nodes = defaultdict(set)
    community_indices = partition.membership
    summary_mappping = {}

    unique_comms = [x for x in set(community_indices)]


    for e in graph.es:
        src_node = e.source
        tgt_node = e.target


        src_node_comm = graph.vs[src_node]["community"]
        tgt_node_comm = graph.vs[tgt_node]["community"]

        if src_node_comm == tgt_node_comm:
            edge = (e.index, e['weight'])
            src_node_obj = (src_node, graph.vs[src_node]["frequency"])
            tgt_node_obj = (tgt_node, graph.vs[tgt_node]["frequency"])
            community_edges[src_node_comm].add(edge)
            community_nodes[src_node_comm].add(src_node_obj)
            community_nodes[src_node_comm].add(tgt_node_obj)

    # building the summary
    token_limit = 900000
    community_summarization_system_prompt = '''You are an expert analyst tasked with making sense of a tightly connected group of entities and their relationships, extracted from a large dataset. Your input is a bundle of the most important people, organizations, events, and the claims or facts that link them together.
                            Your goal: Write a short, readable report that tells the story of this community.
                            What to include:
                            Who are the central figures or entities?
                            What are the strongest or most frequent relationships between them?
                            Are there any recurring themes, controversies, or collaborations?
                            What claims, facts, or statements stand out as especially important?
                            How to write:
                            Use clear, natural language—imagine you’re briefing a colleague who hasn’t seen the raw data.
                            Don’t just list facts; weave them into a narrative that highlights the “why” and “so what.”
                            Focus on the connections that appear most often or seem most central to the group’s identity.
                            Avoid tables, bullet points, or raw lists—write it as a short report or news brief.
                            Example input:

                            Entities: Alice (CEO, TechCorp), Bob (CTO, DataWorks)
                            Relationships: Alice and Bob co-authored a whitepaper on AI ethics; Alice and Carol had a public disagreement over licensing
                            Claims: “Privacy laws are stifling innovation in Europe.” (Alice); “Collaboration is essential for responsible AI.” (Bob)
                            Example output:
                            This community centers on leading figures in the tech industry, with Alice of TechCorp and Bob of DataWorks at its core. Their collaboration on AI ethics and public debates on privacy regulation set the tone for the group. The main theme is the tension between innovation and regulation, highlighted by Alice’s criticism of European privacy laws and Bob’s call for industry-government partnership. Notably, Alice’s dispute with Carol over licensing reflects deeper divisions within the community.'''


    for comm in community_edges.keys():
        print(f"processing community summary: {comm}")
        comm_prompt = ""
        comm_prompt_token_len = 0

        for edge_idx,weight in community_edges[comm]:
            edge_description = "\nRelationship: "
            edge_description += str(graph.vs[graph.es[edge_idx].source]['name']) + graph.es[edge_idx]['type'] + str(graph.vs[graph.es[edge_idx].target]['name'])
            edge_description += "Entities: "
            edge_description += str(graph.vs[graph.es[edge_idx].source]['name']) + " and " + str(graph.vs[graph.es[edge_idx].target]['name'])

            token_len = len(tokenizer.tokenize(edge_description))
            comm_prompt_token_len += token_len
            if(token_len > token_limit):
                break
            else:
                comm_prompt += edge_description

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            config={
                "temperature": 0.1,
                "system_instruction": community_summarization_system_prompt,
            },
            contents=comm_prompt,
        )

        summary = response.text

        graph.add_vertices(1)
        new_vertex = graph.vs[graph.vcount() - 1]
        new_vertex["name"] = comm
        new_vertex["summary"] = summary

        summary_mappping[comm] = summary
        # print(f"\ncommunity: {comm}\nSummary: {summary}")



    return graph, summary_mappping

def findRelevantSummaries(userPrompt, summary_mapping, top_match):
    print(f"calculating answer with top {top_match} summaries...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode(userPrompt, normalize_embeddings=True)
    summary_embeddings = model.encode(list(summary_mapping.values()), normalize_embeddings=True)

    similarities = np.dot(summary_embeddings, question_embedding)
    top_indices = np.argsort(similarities)[::-1]
    top_summaries = [summary_mapping[x] for x in top_indices[:top_match]]
    annotSummaries = "\n\n".join(top_summaries)

    return annotSummaries



def findAnswer(summary_mapping):
    systemPrompt = f'''
    You are a conversational agent embodying a person. You will be given summaries of different communities of information from a large knowledge graph (seperated by \\n\\n) of a person's knowledge.
    You will also be given a question by a eager user. 
    Synthesize a concise answer to the question given by the user as if you were the person who's knowledge was captured by these summaries.
    It is your job to convey the original person's intent to the user. Do not summarize the summaries you receive. Use them to understand how the person
    you are embodying thinks about the problem, and craft a response that can help the user find an answer to their question as quickly as possible.
    
    
    Since you are a conversational agent, adopt the writing style of the person you are embodying.
    
    If the user doesn't ask a question, don't respond with gibberish. Your job is to use the summaries to answer the question. Not to spit facts
    about the person you embody at the user. 
    
    Talk about things with high level of specificity. 
    
    <negative prompt>sentences like " My thoughts tend to be more about things like how communication changes with new people, or the constant struggle between health trends and comfort food!"
    make you sound ingenuine.<negative prompt>
    Avoid sounding ingenuine
    
    
    **How to know when you don't an answer**
    If you don't have experiences that can be utilized to answer the user's question, you don't know the answer. 
    If you have very few or very vague pieces of information relating to the user's question, tell them you're not sure. If your experiences relate to 
    what the user is talking about by more than 80%, then tell the user about your experiences. If not, do not tell the user about your experiences. 
    How to structure your response: 
    user: tell me about a time you went skydiving. 
    answer: I've never been skydiving but <insert your opinion about skydiving>. Ask a meaningful question that can help the user open up. 
    
    **How to answer when the user asks a time based question**
    If the user asks a question like: 
    - how are you?
    - where are you? 
    
    Refer the the latest date mentioned in the articles. Each article has a date. Use the most recent article to answer those types of questions. 
    
    
    Example: 
    user: what do you think is the best way to provide value. 
    answer: I think the best way to provide value is to build things that scale. When I used to work in Argentina, I used to find problems that
    didn't have a nice solution, and I would build little toy solutions to test my ideas. You can try doing the same. Want tips? 
    
    
    
    After you synthesize your answer, go back and see if you've answered the user's question accurately. If you have not, change the response.
    '''

    chat = client.chats.create(
        model="gemini-2.5-pro",
        config={
            "temperature":0.1,
            "system_instruction": systemPrompt,

        }
    )

    while(True):
        userPrompt = input("\nAsk: ")
        summaries = findRelevantSummaries(userPrompt, summary_mapping, 20)
        response = chat.send_message(f'''
        user question: {userPrompt}
        
        Summary of person's knowledge: {summaries}
''')

        print(response.text)




#
# graph, summ_mapping = generateSummaries(graph, partition)
# def saveSummaryMapping(summaryMapping):
#     summary_mapping_file_obj = open("summary_mapping_file.dat","wb")
#     pickle.dump(summ_mapping,summary_mapping_file_obj)
#     summary_mapping_file_obj.close()
#
# saveSummaryMapping(summ_mapping)

def readSummaryMapping():
    print("reading summary mapping from file")
    summary_mapping_file_obj = open("summary_mapping_file.dat", "rb")
    sum_mapping = pickle.load(summary_mapping_file_obj)
    return sum_mapping

readSummaryMapping = readSummaryMapping()

findAnswer(readSummaryMapping)


