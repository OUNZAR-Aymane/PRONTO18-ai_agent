from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mistralai import Mistral
from unstructured.partition.pdf import partition_pdf
from Embedder import Embedder

class MultimodalEmbedder(Embedder):
    def __init__(self,api_key):
        super().__init__(api_key)
    
    def summarize_Table(self,item):
        """
            Summarize a table or text chunk using the provided summarize_chain.
        """
        # Create a new instance of the ChatMistralAI model
        text_model = ChatMistralAI(mistral_api_key=self.api_key,model="mistral-large-latest")
        # Define the prompt for summarization
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.
        The summary must be in french.
        Respond only with the summary, no additionnal comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        Optimize your summary embedding as it will be used for RAG.
        Table or text chunk: {element}
        """
        # Create a prompt template
        prompt = ChatPromptTemplate.from_template(prompt_text)
        # Create a chain that combines the prompt, model, and output parser
        summarize_chain = {"element": lambda x: x} | prompt | text_model | StrOutputParser()
        # Invoke the chain with the item and return the summary
        return summarize_chain.invoke({"element": item})


    def summarize_image(self,b64,prefix,suffix):
        """
        Summarize an image using the Mistral model.
        Args:
            b64 (str): Base64 encoded image string.
            prefix (str): Text before the image.
            suffix (str): Text after the image.
        Returns:
            str: Summary of the image.
        """
        # Create a new instance of the Mistral model
        client = Mistral(api_key=self.api_key) 
        # Define the prompt for summarization
        prompt = f""" You are given:
        - an inline image (base-64) extracted from a PDF
        - the French text immediately before and after the image

        Text-before:
        {prefix}

        Text-after:
        {suffix}

        Task:
        - Write **one paragraph (≤ 150 words, in French)** that describes only what is visible in the image.
        - If the image is the logo of "Association Gens de la lune", do not describe it.
        - Use the surrounding text only to resolve names, labels or context; do not repeat or paraphrase it.
        - Keep the prose concise and optimised for semantic retrieval (RAG).
        - Your priority is to describe the image, not to summarize the text.
        - Do not mention that you are describing an image, and do not start with phrases like “Cette image montre”.
        - Do not include any additional information or context that is not visible in the image.
        - If the image is not relevant to the text, do not describe it.
        Respond in French.  Output only the paragraph."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]
        
        # Perform inference
        response = client.chat.complete(
            model="pixtral-large-latest",
            messages=messages,
        )
        # Return the model's output
        return response.choices[0].message.content


    
    def image_wrapper(self,core_elems, text_list, i, max_chars=200, elements_nb=3):
        """
        Create context for an image by extracting text before and after it.
        
        Args:
            core_elems: List of document elements
            text_list: List of previously processed text elements
            i: Current index in core_elems
            max_chars: Maximum number of characters to include in prefix/suffix
            elements_nb: Number of elements to consider for context
        
        Returns:
            tuple: (prefix, suffix) containing text before and after the image
        """
        # Construct the prefix from the last `elements_nb` items in `text_list`
        prefix = " \n".join(text_list[-elements_nb:])
        prefix = prefix[-max_chars:]  # Truncate to `max_chars` characters

        # Construct the suffix from the next `elements_nb` elements in `core_elems`
        suffix = [
            el.text for el in core_elems[i:i + elements_nb]
            if el.category not in {"Image", "Table"}
        ]
        suffix = " \n".join(suffix)
        if len(suffix) > max_chars:
            suffix = suffix[:max_chars]  # Truncate to `max_chars` characters

        return prefix, suffix

    def get_core_elements(self,file_path):
        # Extract elements from PDF with table and image extraction enabled
        elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,            # extract tables
            strategy="hi_res",                     # mandatory to infer tables
            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will be saved in base64
            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage
        )

        # Filter out headers, footers, and page numbers
        core_elems = [
            el for el in elements
            if el.category not in ("Header", "Footer", "PageNumber")
        ]
        return core_elems
    
    def load_and_split(self,file_path):
        """
        Process document elements, handle images and tables, and split into chunks.
        """
        core_elems = self.get_core_elements(file_path)  # Extract core elements from the PDF
        # Initialize a text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text = ""  # Initialize an empty string to store the combined text
        text_list = []  # List to keep track of processed text elements
        # Iterate through each element in the core elements
        for i, element in enumerate(core_elems):
            if element.category == "Image":
                # If the element is an image, extract its base64 representation
                b64 = element.metadata.image_base64
                # Generate prefix and suffix for the image using the wrapper function
                prefix, suffix = self.image_wrapper(core_elems, text_list, i)
                # Summarize the image using the summarize_image function
                image_summary = self.summarize_image(b64, prefix, suffix)
                # Append the image summary to the combined text
                text += image_summary + "\n"
            elif element.category == "Table":
                # If the element is a table, extract its HTML representation
                html = element.metadata.text_as_html
                # Summarize the table using the summarize_Table function
                table_summary = self.summarize_Table(html)
                # Append the table summary to the combined text
                text += table_summary + "\n"
                # Add the table summary to the text list for context
                text_list.append(table_summary)
            else:
                # For other elements, append their text content to the combined text
                text += element.text + "\n"
                # Add the text content to the text list for context
                text_list.append(element.text)

        # Wrap the full combined text into a Document object
        document = Document(page_content=text)

        # Split the document into smaller chunks using the text splitter
        chunks = text_splitter.split_documents([document])
        return chunks

            

