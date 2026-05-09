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
            "long_term": [],
            "episodic_log": [],
            "self_knowledge": {
                "architecture_facts": [],
                "failure_patterns": [],
                "recovery_methods": []
            },
            "filesystem_map": {}
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

    def get_smart_context(self, query: str, q_emb, episodic_embeddings: list) -> str:
        """
        Smart retrieval: last 3 USER episodic entries + top 2 semantic hits.
        Vault always included. Deduplicates overlaps.
        Autonomous system logs ([AUTONOMOUS] prefix) are excluded from retrieval —
        they pollute user query context with irrelevant system activity.
        q_emb: pre-computed CPU tensor from agent._encode(); avoids calling miniLM here.
        """
        import torch

        profile   = self.memory.get("user_profile", {})
        project   = self.memory.get("project_state", {})
        long_term = self.memory.get("long_term", [])
        episodes  = self.memory.get("episodic_log", [])

        # Build a filtered index map: only user-facing interactions
        # autonomous entries start with "[AUTONOMOUS]" or "[TASK FAILED]" or "[TASK RETRY]"
        SYSTEM_PREFIXES = ("[AUTONOMOUS]", "[TASK FAILED]", "[TASK RETRY]")
        user_indices = [
            i for i, ep in enumerate(episodes)
            if not ep.get("summary", "").startswith(SYSTEM_PREFIXES)
        ]

        selected_indices = set()

        # 1. Last 3 user entries by recency
        if user_indices:
            last3 = user_indices[-3:]
            for idx in last3:
                selected_indices.add(idx)

        # 2. Top 2 semantic hits — only over user entries
        if user_indices and episodic_embeddings and len(episodic_embeddings) == len(episodes):
            user_embs = torch.stack([episodic_embeddings[i] for i in user_indices])
            cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), user_embs)
            top_k = min(2, len(user_indices))
            top_local = cos_sims.topk(top_k).indices.tolist()
            for local_idx in top_local:
                selected_indices.add(user_indices[local_idx])

        # 3. Build context string
        context = "--- MEMORY CONTEXT ---\n"

        # Identity
        context += f"USER: {profile.get('name', 'Unknown')} | ROLE: {profile.get('role', 'User')}\n"
        context += f"TECH STACK: {', '.join(profile.get('preferences', {}).get('tools', []))}\n"
        context += f"CURRENT PHASE: {project.get('current_phase', 'Unknown')}\n"

        # Response style preference — injected only when non-default
        response_style = profile.get('preferences', {}).get('response_style', 'default')
        style_note = profile.get('preferences', {}).get('style_note', '')
        if response_style and response_style != 'default':
            context += f"RESPONSE STYLE: {response_style}"
            if style_note:
                context += f" ({style_note})"
            context += "\n"

        # Known file system locations
        env = profile.get('environment', {})
        known_locations = env.get('known_locations', {})
        if known_locations:
            context += "\n[KNOWN LOCATIONS]:\n"
            for label, path in known_locations.items():
                context += f"- {label}: {path}\n"

        # Self knowledge — CLARA's operational learnings about her own architecture
        sk = self.memory.get('self_knowledge', {})
        sk_lines = []
        for fact in sk.get('architecture_facts', []):
            if fact.get('status') == 'active':
                sk_lines.append(f"  [ARCH] {fact['summary']} — {fact['detail']} [conf:{fact['confidence']}]")
        for pat in sk.get('failure_patterns', []):
            if pat.get('status') == 'active':
                sk_lines.append(f"  [FAIL] Trigger: {pat['trigger']} | Fix: {pat['correct_approach']}")
        for rec in sk.get('recovery_methods', []):
            if rec.get('status') == 'active':
                sk_lines.append(f"  [RECV] {rec['problem']} → {rec['method']}")
        if sk_lines:
            context += "\n[SELF KNOWLEDGE — CLARA's operational learnings. CLAUDE.md takes precedence on all architectural matters]:\n"
            context += "\n".join(sk_lines) + "\n"

        # Filesystem map — progressively discovered directory/file tree
        fsmap = self.memory.get('filesystem_map', {})
        if fsmap:
            context += "\n[FILE SYSTEM MAP]:\n"
            context += self._serialize_filesystem_map(fsmap)

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

        # Log the full context being passed to Grok — file only, not console
        # Skipping logging for now for verboiseness
        # try:
        #     import logging
        #     flog = logging.getLogger("clara_session")
        #     # Only log to file handlers to keep terminal clean
        #     file_handlers = [h for h in flog.handlers
        #                      if isinstance(h, logging.FileHandler)]
        #     if file_handlers:
        #         record = logging.LogRecord(
        #             name="clara_session", level=logging.DEBUG,
        #             pathname="", lineno=0,
        #             msg=f">> [MEMORY_CONTEXT] Injecting into Grok:\n{context}",
        #             args=(), exc_info=None
        #         )
        #         for h in file_handlers:
        #             h.emit(record)
        # except Exception:
        #     pass

        return context

    def update_response_style(self, style: str, note: str = "") -> None:
        """
        Update Alkama's response style preference.
        Called from memorize_episode when a style_update is extracted.
        style: e.g. "concise", "detailed", "default"
        note: brief reason e.g. "Alkama said responses were too verbose"
        """
        if "preferences" not in self.memory["user_profile"]:
            self.memory["user_profile"]["preferences"] = {}
        self.memory["user_profile"]["preferences"]["response_style"] = style
        self.memory["user_profile"]["preferences"]["style_note"] = note
        self._save_memory()

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
        print(f"   [Memory] Logged to Stream: {summary[:50]}...")

    def add_episodic_entry(self, summary: str, encode_callback=None):
        """
        Unified episodic write — always writes log entry.
        If encode_callback is provided, also encodes and returns the embedding
        so the caller can append it to episodic_embeddings atomically.
        encode_callback: callable(summary: str) → CPU tensor
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }
        self.memory["episodic_log"].append(entry)
        self._save_memory()

        if encode_callback is not None:
            try:
                return encode_callback(summary)
            except Exception as e:
                print(f"   [Memory] Embedding encode failed: {e}")
        return None

    def add_long_term_fact(self, fact):
        """
        Saves a permanent fact to the Vault. Exact string dedup guard.
        """
        if fact in self.memory["long_term"]:
            return
        self.memory["long_term"].append(fact)
        self._save_memory()
        print(f"   [Memory] 🔒 Locked to Vault: {fact}")

    def add_self_knowledge(self, category: str, entry: dict) -> bool:
        """
        Add a self_knowledge entry with dedup guard.
        category: 'architecture_facts' | 'failure_patterns' | 'recovery_methods'
        Returns True if added, False if duplicate.
        """
        if 'self_knowledge' not in self.memory:
            self.memory['self_knowledge'] = {
                'architecture_facts': [], 'failure_patterns': [], 'recovery_methods': []
            }
        cat_list = self.memory['self_knowledge'].setdefault(category, [])
        dedup_key = {
            'architecture_facts': 'summary',
            'failure_patterns': 'trigger',
            'recovery_methods': 'problem'
        }.get(category, 'summary')
        new_val = entry.get(dedup_key, '').lower().strip()
        for existing in cat_list:
            if existing.get(dedup_key, '').lower().strip() == new_val:
                return False
        cat_list.append(entry)
        self._save_memory()
        from .session_logger import slog
        slog.info(f">> [Self-Knowledge] Added to {category}: {entry.get(dedup_key, '')[:80]}")
        return True

    def merge_filesystem_path(self, path_str: str, is_file: bool = False) -> None:
        """
        Add a confirmed file or directory path into the filesystem_map tree.
        path_str: absolute Windows path e.g. 'E:\\ML PROJECTS\\AGENT_ZERO\\api.py'
        is_file: True for a file node (value=None), False for a directory node (value={}).
        Existing nodes are never overwritten — only new nodes are added.
        """
        if 'filesystem_map' not in self.memory:
            self.memory['filesystem_map'] = {}
        path_str = path_str.replace('/', '\\').rstrip('\\')
        parts = [p for p in path_str.split('\\') if p]
        if not parts:
            return
        drive = parts[0].rstrip(':')
        rest = parts[1:]
        if not drive or not rest:
            return
        node = self.memory['filesystem_map']
        if drive not in node:
            node[drive] = {}
        node = node[drive]
        for i, part in enumerate(rest):
            is_last = (i == len(rest) - 1)
            if is_last:
                if part not in node:
                    node[part] = None if is_file else {}
            else:
                if part not in node or node[part] is None:
                    node[part] = {}
                node = node[part]
        self._save_memory()

    def remove_filesystem_path(self, path_str: str) -> None:
        """
        Remove a stale entry from the filesystem_map when confirmed not found.
        """
        path_str = path_str.replace('/', '\\').rstrip('\\')
        parts = [p for p in path_str.split('\\') if p]
        if not parts:
            return
        drive = parts[0].rstrip(':')
        rest = parts[1:]
        fsmap = self.memory.get('filesystem_map', {})
        node = fsmap.get(drive)
        if node is None:
            return
        for part in rest[:-1]:
            if not isinstance(node, dict) or part not in node:
                return
            node = node[part]
        if isinstance(node, dict) and rest and rest[-1] in node:
            del node[rest[-1]]
            self._save_memory()

    def _serialize_filesystem_map(self, fsmap: dict, indent: int = 0) -> str:
        """
        Compact human-readable serialization of the filesystem_map tree for context injection.
        Directories are shown with trailing backslash; unexplored dirs labeled [unexplored].
        Files in a directory are grouped inline, up to 8 per line with [+N more] overflow.
        """
        MAX_FILES_INLINE = 8
        lines = []
        prefix = '  ' * indent
        for key, val in sorted(fsmap.items()):
            if val is None:
                # file — collected by parent, not emitted here individually
                continue
            if isinstance(val, dict):
                # Drive letters (single char at top level) shown with colon
                display_key = f"{key}:" if (indent == 0 and len(key) == 1) else key
                files = sorted(k for k, v in val.items() if v is None)
                subdirs = {k: v for k, v in val.items() if isinstance(v, dict)}
                if not files and not subdirs:
                    lines.append(f"{prefix}{display_key}\\  [unexplored]")
                else:
                    lines.append(f"{prefix}{display_key}\\")
                    if files:
                        shown = files[:MAX_FILES_INLINE]
                        line = f"{prefix}  " + '  '.join(shown)
                        if len(files) > MAX_FILES_INLINE:
                            line += f"  [+{len(files) - MAX_FILES_INLINE} more]"
                        lines.append(line)
                    if subdirs:
                        lines.append(self._serialize_filesystem_map(subdirs, indent + 1).rstrip('\n'))
        return '\n'.join(lines) + '\n'