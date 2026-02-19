from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

def get_token_nodes(documents: list[Document]):  
    character_splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=0,
        paragraph_separator="\n\n",
    )

    token_splitter = TokenTextSplitter(
        chunk_size=256,     
        chunk_overlap=20,   
    )

    sentence_nodes = character_splitter.get_nodes_from_documents(documents)
    token_nodes = []
    for i, node in enumerate(sentence_nodes):
        node_text = node.get_content()
        temp_doc = Document(text=node_text, metadata=node.metadata)
        sub_nodes = token_splitter.get_nodes_from_documents([temp_doc])
        for sub_node in sub_nodes:
            sub_node.metadata.update(node.metadata)
        
        token_nodes.extend(sub_nodes)
    
    return token_nodes
