from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from inference.CoT.utils import *
from config import *
from inference.CoT.struct_output import *
from beam import *

llm = ChatOpenAI(
	model=MODEL_NAME,
	openai_api_key="EMPTY",
	max_tokens=512,
	openai_api_base=INFERENCE_SERVER_URL,
	temperature=0,
	streaming= False
)
tokenizer = AutoTokenizer(MODEL_NAME)

def check_grade_lst(question, text):
	prompt_text_grader = PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
		of a list of retrieved passages to a user question. The goal is to filter out erroneous retrievals. \n
		Return only the passage index whether the passage is relevant to the question. \n
		Provide the output as a JSON with passage index seperated by a comma and no premable or explaination.
		 <|eot_id|><|start_header_id|>user<|end_header_id|>
		Here is the list of retrieved text: \n\n {text} \n\n
		Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
		""",
		input_variables=["question", "text"]
	)
	structured_llm_grader = llm.with_structured_output(GradeRelationList)
	relation_grader = prompt_text_grader | structured_llm_grader 
	result = relation_grader.invoke({"question": question, "text": text})
	return result

def contextual_question_retrieval(claims):
	system_promt=("You are a helpful assistant responsible for generating a comprehensive summary of the data provided below."
	" Given the list of claims that may relation with each other. Please write a Concise summary of claims that aim to provide a contextual information."
	" The output just generate a concise summary without any explaination."
	" Please note that if the provided claims are contradictory, please resolve the contradictions and provide a single, coherent summary (no need Here is part)")
	chat_template_contextual = tokenizer.apply_chat_template(
		[
			{"role":"system", "content":"{system}"},
			{"role":"user", "content":"\nHere is the list of claims {claims}\n"}
		], tokenize=False, add_generation_prompt=True)
	
	prompt_summary_contextual = PromptTemplate(template=chat_template_contextual, input_variables=["system", "claims"])
	structured_summary_contextual = llm.with_structured_output(ContextualOutput)
	contextual_chain = prompt_summary_contextual | structured_summary_contextual 
	results = contextual_chain.invoke({"system": system_promt, "claims": claims})
	return results

def check_response(question, context):
	system_promt=("You are an advanced AI assistant skilled in analyzing textual data."
		"\nBelow is a question and relevant passages that may contain information to answer it."
		"\nYour task is to determine if the provided passages contain enough relevant information to answer the question, even if not directly stated."
		"\nConsider both direct answers and implied or partially inferred information."
		"\nReturn a binary score: 'True' if the context provides sufficient information to answer the question; 'False' if it does not."
		"\nProvide only the binary score in JSON format with a single key 'score'. Do not include explanations.")
	
	chat_template_check = tokenizer.apply_chat_template(
		[
			{"role":"system", "content":"{system_promt}"},
			{"role":"user", "content":"\nQuestion: {question}\nRelevan Passages: {context}"}
		], tokenize=False, add_generation_prompt=True)
	
	prompt_check_response = PromptTemplate(template=chat_template_check, input_variables=["system_promt", "question","context"])
	structured_check_content= llm.with_structured_output(GradeResponse)
	check_response_chain = prompt_check_response | structured_check_content 
	results = check_response_chain.invoke({"system_promt": system_promt, "question": question ,"context": context})
	return results

def gen_question(question, context, previous_though):
	system_promt_gen_answer = (
		"You are an advanced AI skilled in generating a concise insightful chain-of-thought query to guide further research and exploration."
		" Below is an input question and relevant context information and previous failed queries."
		"\nYour task is to :"
		"\n1. Analyze the input question to understand its intent and identify gaps in the provided context that prevent a complete answer."
		"\n2. Generate a new chain-of-thought query that is based on the input question, incorporating logical steps or deeper aspects of the topic."
		" This new query should be designed to guide further search or inquiry, aiming to bridge the identified gaps and refine the search for an answer."
		"\n3. Avoid repeating or rephrasing any of the previous failed queries. Instead, aim to expand the scope or explore different facets of the topic that have not been addressed yet."
		"All JSON MUST in correct format"
		"**DO NOT get information from 'Relevant context information' to create new input variables.**"
	)

	chat_gen_answer = tokenizer.apply_chat_template(
		[
			{"role": "system", "content": system_promt_gen_answer},
			{"role": "user", "content": f"\nQuestion: {question}\nRelevant context information: {context}\nPrevious failed queries: {previous_though}"}
		],
		tokenize=False, 
		add_generation_prompt=True
	)
	
	prompt_gen_answer = PromptTemplate(template=chat_gen_answer, input_variables=["system_promt_gen_answer", "question", "context", "previous_though"])
	structured_check_content = llm.with_structured_output(GenQuery)
	chain_gen_answer = prompt_gen_answer | structured_check_content
	answer = chain_gen_answer.invoke({"system_promt_gen_answer": system_promt_gen_answer, "question": question, "context": context, "previous_though": previous_though})

	return answer


def final_answer(question, context):
	system_promt_gen_answer=("You are an expert AI designed to analyze information from retrieval-augumented generation system."
	"\nYour task is to answer questions based on the input context. Below is a question along with the input context."
	"\nMake sure your repsonse is consice clear, and directly answer the question in maximum 5 sentences WITHOUT any explaination."
	"\nDO NOT use any external knowledge. "
	"\nIf the answer is not directly found, try to infer the best possible answer from the context.")
	
	chat_gen_answer= tokenizer.apply_chat_template(
		[
			{"role":"system", "content":"{system_promt_gen_answer}"},
			{"role":"user", "content":"\nQuestion: {question}\nInput context: {context}"}
		], tokenize=False, add_generation_prompt=True)
	prompt_gen_answer = PromptTemplate(template=chat_gen_answer, input_variables=["system_promt_gen_answer", "question","context"])
	chain_gen_answer = prompt_gen_answer | llm | StrOutputParser()
	answer = chain_gen_answer.invoke({"system_promt_gen_answer": system_promt_gen_answer,"question":question, "context": context}).strip()
	return answer


