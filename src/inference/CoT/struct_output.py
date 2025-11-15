from langchain_core.pydantic_v1 import BaseModel, Field

class GradeRelation(BaseModel):
    """Binary score for relevance check on retrieved text."""
    binary_score: str = Field(
        description="The Text is relevant to the question, 'yes' or 'no'"
    )
	
class GradeRelationList(BaseModel):
	"""List passage index check on retrieved text."""
	passage_idx: str = Field(
		description="The passage index of relevant chunks, seperated by a comma"
	)

class ContextualOutput(BaseModel):
	"""contextual summarization for the input question."""
	summary: str = Field(
		description="Concise summary ocontextual information of the input question"
	)

class ContextualTriplets(BaseModel):
	"""contextual generation of knowledge subgraph."""
	context: str = Field(
		description="generate concise contextual information based on list of triplets."
	)

class GradeResponse(BaseModel):
	"""Binary score to determine if the passages provide sufficient information to answer the question directly."""
	binary_score: bool = Field(
		description="The relevant passages provide sufficient information to answer the question directly, 'yes' or 'no'"
	)

class GenQuery(BaseModel):
	"""Generate  chain-of-thought query for futher research and exploration."""
	new_query: str = Field(
		description="Generate new chain-of-thought query for futher research and exploration"
	)
	