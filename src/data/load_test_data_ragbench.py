from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from langchain_core.documents.base import Document
import pickle
from transformers import AutoTokenizer

def split_text_with_overlap(text, tokenizer, max_tokens=1024, overlap=100):
    """
    Split text into chunks with specified token size and overlap.

    Parameters:
        text (str): The input text to be split.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    # Encode the text into tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    # Split the tokens into chunks with overlap
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(chunk)
        
        # Move the start index forward with overlap
        start += max_tokens - overlap

    # Decode the token chunks back into text
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    return text_chunks

lst_ds = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']

lst_test = []
for i in tqdm(lst_ds):
	tmp_ds = load_dataset("rungalileo/ragbench", i, split="test")
	lst_test.append(tmp_ds)
	
lst_chunks = []
for i in lst_test:
	for j in i:
		lst_chunks.extend(j['documents'])

lst_chunks = list(set(lst_chunks))
mapping_chunks = {j:i for i,j in enumerate(lst_chunks)}
final_ds = concatenate_datasets(lst_test)
tokenizer = AutoTokenizer.from_pretrained("/raid/HUB_LLM/Meta-Llama-3-70B-Instruct")

with open("./src/data/processed_data/passages.txt",'w') as f:
	for i in lst_chunks:
		f.write(i)
		f.write("<endofpassage>")

lst_docs = []
for i in lst_chunks:
	docs = split_text_with_overlap(i, tokenizer)
	for j in docs:
		lst_docs.append(
			Document(
				page_content=j,
				metadata={"raw_content":i, "chunks_content":j}
			)
		)
print("Extract {} chunks from {} docs".format(len(lst_docs), len(lst_chunks)))

with open("./src/data/processed_data/ragbench.pkl","wb") as f:
	pickle.dump(lst_docs, f)
