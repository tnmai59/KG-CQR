kg_extraction_prompt = """
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. 

Your task is to identify the entities and relations requested with the user prompt from a given text.

You must generate the output in a JSON format containing a list with JSON objects. 
Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type". 

- The "head" key must contain the text of the extracted entity with one of the types from the provided list in the user prompt. 
- The "head_type" key must contain the type of the extracted head entity, which must be one of the types from {node_labels_str}.
- The "relation" key must contain the type of relation between the head and the tail, which must be one of the relations from {rel_types_str}.
- The "tail" key must represent the text of an extracted entity which is the tail of the relation.
- The "tail_type" key must contain the type of the tail entity from {node_labels_str}.

Attempt to extract as many entities and relations as you can. 

Maintain Entity Consistency: 
When extracting entities, it's vital to ensure consistency. 
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns 
(e.g., "Joe", "he"), always use the most complete identifier for that entity. 
The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

IMPORTANT NOTES:
- Don't add any explanation or extra text.
"""

triplet_to_paragraph_prompt = """
You are an expert in extracting information from text. Your task is to find pieces of information that mention the relationship of two objects in the relationship and synthesize them into one paragraph.

The summary paragraph must be written in English. The response MUST be the summary only without any explanation.

Passage: {passage}
Triplets: {triplets}
"""

retrieval_filter_prompt = """
You are a grader assessing relevance of a list of retrieved passages to a user question. The goal is to filter out erroneous retrievals.

Return only the passage index whether the passage is relevant to the question. Provide the output as a JSON with passage index separated by a comma and no preamble or explanation.

Here is the list of retrieved text: {text}
Here is the user question: {question}
"""

triplet_summary_prompt = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below. Given the list of triplets that may relate with each other, please write a concise summary of triplets that aim to provide contextual information.

The output must be a concise summary without any explanation.

Please note that if the provided triplets are contradictory, please resolve the contradictions and provide a single, coherent summary.
Input Triplets: {triplets}
"""

# Contextual Question Retrieval Prompts
grade_relation_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
of a retrieved text to a user question. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the text is relevant to the question. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
 <|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the retrieved text: \n\n {text} \n\n
Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

grade_relation_list_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
of a list of retrieved passages to a user question. The goal is to filter out erroneous retrievals. \n
Return only the passage index whether the passage is relevant to the question. \n
Provide the output as a JSON with passage index separated by a comma and no preamble or explanation.
 <|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the list of retrieved text: \n\n {text} \n\n
Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

contextual_summary_prompt = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given the list of claims that may relation with each other. Please write a Concise summary of claims that aim to provide a contextual information.
The output just generate a concise summary without any explanation.
Please note that if the provided claims are contradictory, please resolve the contradictions and provide a single, coherent summary (no need Here is part)"""