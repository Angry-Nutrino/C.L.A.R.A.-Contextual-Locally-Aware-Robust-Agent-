from sentence_transformers import SentenceTransformer
import torch

def expand_query(q):
    expansions = {
        "price": ["price", "rate", "cost", "value", "how much"],
        "gold": ["gold", "gold rate", "gold price today", "gold per gram"],
        "stock": ["stock", "share", "stock price", "ticker", "quote"],
        "phone number": ["phone number", "contact number", "mobile", "email"],
        # add more as you test
    }
    words = q.lower().split()
    expanded = q
    for word in words:
        if word in expansions:
            expanded += " " + " ".join(expansions[word])
    return expanded



def miniLm_pre_compute():
        # Load model once at startup
        miniLM = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
        tools = [
                {
                        "name": "web_search",
                        "description": "search internet for current facts, real-time prices, stock quotes, cryptocurrency rates, gold price, commodity values, news, events, company information",
                        "sub_descriptions": [
                        "get latest stock price AAPL TSLA or any ticker",
                        "current gold silver oil price per ounce gram",
                        "find up-to-date news or facts that require external data",
                        "search anything on the web including phones electronics metals"
                        ]
                },
                {
                        "name": "python_repl",
                        "description": "execute python code for calculations math data processing chart generation unit conversions statistics",
                        "sub_descriptions": [
                        "perform mathematical computations or solve equations",
                        "analyze small datasets plot graphs visualize data",
                        "run code snippets that don't require internet access"
                        ]
                },
                {
                        "name": "consult_archive",
                        "description": "search internal company knowledge base document archive resume personal data for relevant information phone number email contact details",
                        "sub_descriptions": [
                        "look for information in resume find phone number email or personal details",
                        "retrieve internal documents reports or knowledge base entries",
                        "get company-specific or archived information not on the web"
                        ]
                }       
        ]

        # Pre-compute: flatten or concatenate descriptions per tool
        tool_embeddings = []
        tool_index_to_name = {}   # to map back later

        for i, tool in enumerate(tools):
                # full_description = tool["description"] 
                sub_embs = miniLM.encode([tool["description"]] + tool["sub_descriptions"], convert_to_tensor=True)
                tool_emb= sub_embs.mean(dim=0)  # max pooling
                # tool_emb = miniLM.encode(full_description, convert_to_tensor=True)
                tool_embeddings.append(tool_emb)
                tool_index_to_name[i] = tool["name"]

        # Stack into one tensor (n_tools, 384)
        tool_embs = torch.stack(tool_embeddings)

        print("Tool embeddings shape:", tool_embs.shape)
        print("Tool index to name mapping:", tool_index_to_name)

        # cosine similarity example
        query = ["I want to know what is the price of gold", "what do you think my height is?", "find the phone number of John Doe in the resume?", "what is the current stock price of Apple?"]
        
        for q in query:
                query_text=expand_query(q)
                q_emb = miniLM.encode(query_text, convert_to_tensor=True)
                cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), tool_embs)
                # best_tool_idx = torch.argmax(cos_sims).item()
                # best_tool_name = tool_index_to_name[best_tool_idx]
                print(f"Query: {q}", "Cosine Similarities:", cos_sims)




if __name__ == "__main__":    miniLm_pre_compute()
    