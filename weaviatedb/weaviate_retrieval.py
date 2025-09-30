
from weaviate.classes.generate import GenerativeConfig

def hybrid_generative_search(client, query, collection_name, grouped_task: str = None, limit: int = 5):
	"""Run a hybrid generative search and return a normalized result."""
	collection = client.collections.get(collection_name)

	if grouped_task is None:
		grouped_task = (
			"Analyze the context and provide brief answers."
			"If certain information is not available in the documents, clearly "
			"indicate what is missing."
		)

	try:
		response = collection.generate.hybrid(
			query=query,
			grouped_task=grouped_task,
			limit=limit,
			generative_provider=GenerativeConfig.openai(model="gpt-4")
		)

		# Normalize absence cases
		if not getattr(response, "objects", None):
			class Dummy:
				class Gen:
					text = (
						"I couldn't find relevant information in the documents to answer that. "
						"Try rephrasing or providing more context."
					)

				generative = Gen()
				objects = []

			return Dummy()

		# Print some debug info (kept minimal)
		try:
			print(f"Search returned {len(response.objects)} relevant documents")
		except Exception:
			pass

		return response

	except Exception as e:
		# Return a small object that the UI can display
		class Err:
			class Gen:
				text = f"Search error: {str(e)}"

			generative = Gen()
			objects = []

		return Err()
