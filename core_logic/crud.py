import json
import os
from datetime import datetime

class crud:
    def __init__(self, filepath="core_logic/memory.json"):
        self.filepath = filepath
        self.memory = self._load_memory()

    def _load_memory(self):
        if not os.path.exists(self.filepath):
            return self._create_default_memory()
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return self._create_default_memory()

    def _create_default_memory(self):
        return {
            "user_profile": {},
            "project_state": {},
            "long_term": [],     # <--- NEW: The "Vault"
            "episodic_log": []   # <--- The "Stream"
        }

    def _save_memory(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save memory: {e}")

    # --- PUBLIC TOOLS ---

    def get_full_context(self):
        """
        DEPRECATED — replaced by get_smart_context(). Kept for reference.
        Fetches the Soul (Profile), The Vault (Long Term), and The Stream (Last 10 interactions).
        """
        profile = self.memory.get("user_profile", {})
        project = self.memory.get("project_state", {})
        long_term = self.memory.get("long_term", [])
        episodes = self.memory.get("episodic_log", [])

        # Get last 10 interactions (The Stream)
        recent_history = episodes[-10:] if len(episodes) > 0 else []

        context = "--- LONG-TERM MEMORY CONTEXT ---\n"
        
        # 1. Identity
        context += f"USER: {profile.get('name', 'Unknown')} | ROLE: {profile.get('role', 'User')}\n"
        context += f"TECH STACK: {', '.join(profile.get('preferences', {}).get('tools', []))}\n"
        
        # 2. Project State
        context += f"CURRENT PHASE: {project.get('current_phase', 'Unknown')}\n"
        
        # 3. The Vault (Permanent Facts)
        if long_term:
            context += "\n[PERMANENT KNOWLEDGE VAULT]:\n"
            for fact in long_term:
                context += f"- {fact}\n"

        # 4. The Stream (Recent History)
        if recent_history:
            context += "\n[RECENT CONVERSATION STREAM (Last 10)]:\n"
            for ep in recent_history:
                context += f"- [{ep.get('timestamp', '')[:16]}] {ep.get('summary', '')}\n"

        context += "--------------------------------"
        return context

    def get_smart_context(self, query: str, miniLM, episodic_embeddings: list) -> str:
        """
        Smart retrieval: last 3 episodic entries + top 2 semantic hits.
        Vault always included. Deduplicates overlaps.
        """
        import torch

        profile   = self.memory.get("user_profile", {})
        project   = self.memory.get("project_state", {})
        long_term = self.memory.get("long_term", [])
        episodes  = self.memory.get("episodic_log", [])

        selected_indices = set()

        # 1. Last 3 by recency
        if episodes:
            last3 = list(range(max(0, len(episodes) - 3), len(episodes)))
            for idx in last3:
                selected_indices.add(idx)

        # 2. Top 2 semantic hits via MiniLM
        if episodes and episodic_embeddings and len(episodic_embeddings) == len(episodes):
            q_emb  = miniLM.encode(query, convert_to_tensor=True).to('cpu')
            all_embs = torch.stack(episodic_embeddings)  # (N, 384) — all on CPU
            cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), all_embs)
            top2_indices = cos_sims.topk(min(2, len(episodes))).indices.tolist()
            for idx in top2_indices:
                selected_indices.add(idx)

        # 3. Build context string
        context = "--- MEMORY CONTEXT ---\n"

        # Identity
        context += f"USER: {profile.get('name', 'Unknown')} | ROLE: {profile.get('role', 'User')}\n"
        context += f"TECH STACK: {', '.join(profile.get('preferences', {}).get('tools', []))}\n"
        context += f"CURRENT PHASE: {project.get('current_phase', 'Unknown')}\n"

        # Vault
        if long_term:
            context += "\n[PERMANENT KNOWLEDGE VAULT]:\n"
            for fact in long_term:
                context += f"- {fact}\n"

        # Selected episodic entries (sorted chronologically)
        if selected_indices:
            context += "\n[RELEVANT PAST INTERACTIONS]:\n"
            for idx in sorted(selected_indices):
                ep = episodes[idx]
                context += f"- [{ep.get('timestamp', '')[:16]}] {ep.get('summary', '')}\n"

        context += "----------------------"
        return context

    def add_episodic_log(self, summary):
        """
        Always saves the summary of the last interaction.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
        self.memory["episodic_log"].append(entry)
        self._save_memory()
        print(f"   [Memory] 📝 Logged to Stream: {summary[:50]}...")

    def add_long_term_fact(self, fact):
        """
        Saves a permanent fact to the Vault.
        """
        if fact not in self.memory["long_term"]:
            self.memory["long_term"].append(fact)
            self._save_memory()
            print(f"   [Memory] 🔒 Locked to Vault: {fact}")