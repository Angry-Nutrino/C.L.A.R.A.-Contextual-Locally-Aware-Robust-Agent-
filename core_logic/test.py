from sentence_transformers import SentenceTransformer
import torch


def miniLm_pre_compute():
        # Load model once at startup
        miniLM = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
        tools = [
                {
                        "name": "web_search",
                        "description": "search the web for prices news and information",
                        "sub_descriptions": [
                        "what is the price of gold",
                        "stock price of Apple AAPL",
                        "latest news about OpenAI",
                        "what is bitcoin trading at",
                        "who won the cricket match",
                        "search online for facts"
                        ]
                },
                {
                        "name": "python_repl",
                        "description": "calculate math and run code",
                        "sub_descriptions": [
                        "calculate the square root of 1445",
                        "convert 45 USD to INR",
                        "plot a bar chart of numbers",
                        "what is 25 percent of 89000",
                        "solve math equation"
                        ]
                },
                {
                        "name": "consult_archive",
                        "description": "search resume CV and knowledge base",
                        "sub_descriptions": [
                        "find the phone number in my resume",
                        "what skills are listed in my CV",
                        "look up project details from the knowledge base",
                        "retrieve contact details from documents",
                        "search archived records"
                        ]
                },
                {
                        "name": "date_time",
                        "description": "get current date and time",
                        "sub_descriptions": [
                        "what is today's date and time",
                        "what time is it right now",
                        "what day is it today"
                        ]
                },
                {
                        "name": "vision_tool",
                        "description": "analyze uploaded image or picture",
                        "sub_descriptions": [
                        "analyze the image I just uploaded",
                        "what is in this picture",
                        "describe the contents of this photo",
                        "identify objects in this image",
                        "tell me what you see in this screenshot"
                        ]
                }
        ]

        # Pre-compute: store ALL sub_description embeddings per tool (for max-similarity)
        tool_all_embs = []       # list of tensors, one per tool, shape (N_subs, 384)
        tool_index_to_name = {}

        for i, tool in enumerate(tools):
                texts = [tool["description"]] + tool["sub_descriptions"]
                embs = miniLM.encode(texts, convert_to_tensor=True)
                tool_all_embs.append(embs)
                tool_index_to_name[i] = tool["name"]

        print(f"Tools loaded: {tool_index_to_name}")
        print(f"Embeddings per tool: {[e.shape[0] for e in tool_all_embs]}")

        # Test queries
        queries = [
                "I want to know what is the price of gold",
                "what do you think my height is?",
                "find the phone number of John Doe in the resume?",
                "what is the current stock price of Apple?",
                "calculate the square root of 144",
                "what time is it right now?",
                "analyze this image for me",
                "who won the IPL match yesterday?",
                "what is 30 percent of 5000?",
                "hello how are you doing?",
        ]

        TOOL_THRESHOLD = 0.40     # above this = tool needed
        NONE_THRESHOLD = 0.15     # below this on ALL tools = no tool needed

        print(f"\n{'='*90}")
        print(f"{'Query':<50} {'Top Tool':<18} {'Score':>6}  {'Verdict'}")
        print(f"{'='*90}")

        for q in queries:
                q_emb = miniLM.encode(q, convert_to_tensor=True)

                # Max-similarity: for each tool, take the BEST matching sub_description
                tool_scores = {}
                for i, embs in enumerate(tool_all_embs):
                        cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), embs)
                        tool_scores[tool_index_to_name[i]] = cos_sims.max().item()

                top_tool = max(tool_scores, key=tool_scores.get)
                top_score = tool_scores[top_tool]

                if top_score < NONE_THRESHOLD:
                        verdict = "NO TOOL (chat)"
                elif top_score >= TOOL_THRESHOLD:
                        verdict = f"-> {top_tool}"
                else:
                        verdict = f"~  {top_tool} (low confidence)"

                q_display = q[:48] + ".." if len(q) > 50 else q
                print(f"{q_display:<50} {top_tool:<18} {top_score:>5.3f}   {verdict}")

        # Detailed breakdown for last query set
        print(f"\n{'='*90}")
        print("DETAILED SCORES (all tools per query):")
        print(f"{'='*90}")
        for q in queries:
                q_emb = miniLM.encode(q, convert_to_tensor=True)
                scores = {}
                for i, embs in enumerate(tool_all_embs):
                        cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), embs)
                        scores[tool_index_to_name[i]] = cos_sims.max().item()

                q_display = q[:48] + ".." if len(q) > 50 else q
                score_str = " | ".join(f"{k}: {v:.3f}" for k, v in scores.items())
                print(f"  {q_display}")
                print(f"    {score_str}")


if __name__ == "__main__":    miniLm_pre_compute()
