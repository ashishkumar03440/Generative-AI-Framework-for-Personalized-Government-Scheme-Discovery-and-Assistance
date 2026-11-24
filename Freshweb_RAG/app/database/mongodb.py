"""MongoDB integration for FreshWeb-RAG using Motor (async driver).

Provides async document collections, indexes, authentication helpers, and
data persistence for users, documents, conversations, memories, and analytics.
Includes an in-memory document store fallback if local MongoDB is not running.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional

import bcrypt

from app.config import settings

logger = logging.getLogger("freshweb_rag")


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


class InMemoryMongoFallback:
    """Lightweight in-memory document database for local testing when MongoDB is offline."""

    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.document_chunks: List[Dict[str, Any]] = []
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []
        self.memory_records: Dict[str, Dict[str, Any]] = {}
        self.query_analytics: List[Dict[str, Any]] = []


class MongoDBManager:
    """Manages connection to MongoDB and handles async CRUD operations."""

    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self.client: Optional[Any] = None
        self.db: Optional[Any] = None
        self.is_connected: bool = False
        self._fallback: InMemoryMongoFallback = InMemoryMongoFallback()

    async def connect(self) -> bool:
        """Connect to MongoDB and ensure indexes with zero-hang fallback."""
        import socket
        from urllib.parse import urlparse

        # Quick pre-flight check if host:port is reachable before launching motor threads
        try:
            parsed = urlparse(self.uri)
            host = parsed.hostname or "127.0.0.1"
            if host.lower() == "localhost":
                host = "127.0.0.1"
            port = parsed.port or 27017
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex((host, port)) != 0:
                    raise ConnectionRefusedError(f"Cannot reach MongoDB socket at {host}:{port}")
        except Exception as sock_err:
            logger.info("MongoDB socket offline (%s). Using resilient in-memory document store.", sock_err)
            self.is_connected = False
            self._fallback = InMemoryMongoFallback()
            return False

        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self.client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=1000,
                connectTimeoutMS=1000,
                socketTimeoutMS=1000,
            )
            # Ping database to verify connection with strict 1s timeout
            await asyncio.wait_for(self.client.admin.command("ping"), timeout=1.0)
            self.db = self.client[self.db_name]
            self.is_connected = True
            logger.info("Connected to MongoDB at %s (database: %s)", self.uri, self.db_name)

            # Ensure indexes
            await self._ensure_indexes()
            return True
        except Exception as exc:
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None
            logger.warning(
                "MongoDB at %s unreachable (%s). Using resilient in-memory store for development.",
                self.uri,
                exc,
            )
            self.is_connected = False
            self._fallback = InMemoryMongoFallback()
            return False

    async def _ensure_indexes(self) -> None:
        """Create indexes on key fields."""
        if not self.is_connected or self.db is None:
            return
        try:
            # Users indexes
            await self.db.users.create_index("email", unique=True)
            await self.db.users.create_index("username", unique=True)

            # Documents indexes
            await self.db.documents.create_index("id", unique=True)
            await self.db.documents.create_index("user_id")

            # Document chunks
            await self.db.document_chunks.create_index("document_id")
            await self.db.document_chunks.create_index("user_id")

            # Conversations indexes
            await self.db.conversations.create_index("id", unique=True)
            await self.db.conversations.create_index("user_id")

            # Conversation messages
            await self.db.conversation_messages.create_index([("conversation_id", 1), ("created_at", 1)])

            # Memory records
            await self.db.memory_records.create_index("id", unique=True)
            await self.db.memory_records.create_index([("user_id", 1), ("memory_type", 1)])

            # Query analytics
            await self.db.query_analytics.create_index([("user_id", 1), ("created_at", -1)])
            logger.info("MongoDB collections and indexes ready.")
        except Exception as exc:
            logger.debug("Failed creating some indexes: %s", exc)

    async def close(self) -> None:
        """Close connection."""
        if self.client:
            self.client.close()
            self.is_connected = False

    # ── User Operations ───────────────────────────────────────────────────────
    async def create_user(self, email: str, username: str, password_hash: str) -> Dict[str, Any]:
        """Create a new user."""
        user_id = str(uuid.uuid4())
        doc = {
            "id": user_id,
            "email": email.strip().lower(),
            "username": username.strip().lower(),
            "password_hash": password_hash,
            "is_active": True,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }
        if self.is_connected and self.db is not None:
            await self.db.users.insert_one(doc)
        else:
            self._fallback.users[user_id] = doc
        return doc

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find a user by email."""
        norm_email = email.strip().lower()
        if self.is_connected and self.db is not None:
            user = await self.db.users.find_one({"email": norm_email}, {"_id": 0})
            return user
        for u in self._fallback.users.values():
            if u.get("email") == norm_email:
                return dict(u)
        return None

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Find a user by username."""
        norm_name = username.strip().lower()
        if self.is_connected and self.db is not None:
            user = await self.db.users.find_one({"username": norm_name}, {"_id": 0})
            return user
        for u in self._fallback.users.values():
            if u.get("username") == norm_name:
                return dict(u)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Find a user by ID."""
        if self.is_connected and self.db is not None:
            user = await self.db.users.find_one({"id": user_id}, {"_id": 0})
            return user
        return self._fallback.users.get(user_id)

    # ── Document Operations ───────────────────────────────────────────────────
    async def insert_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store document record."""
        doc = dict(doc_data)
        doc.setdefault("id", str(uuid.uuid4()))
        doc.setdefault("uploaded_at", datetime.datetime.utcnow().isoformat())
        doc.setdefault("status", "pending")
        if self.is_connected and self.db is not None:
            await self.db.documents.insert_one(dict(doc))
        else:
            self._fallback.documents[doc["id"]] = doc
        return doc

    async def get_document(self, doc_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        q: Dict[str, Any] = {"id": doc_id}
        if user_id:
            q["user_id"] = user_id
        if self.is_connected and self.db is not None:
            return await self.db.documents.find_one(q, {"_id": 0})
        d = self._fallback.documents.get(doc_id)
        if d and (user_id is None or d.get("user_id") == user_id):
            return dict(d)
        return None

    async def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """List all documents for a user."""
        if self.is_connected and self.db is not None:
            cursor = self.db.documents.find({"user_id": user_id}, {"_id": 0}).sort("uploaded_at", -1)
            return await cursor.to_list(length=200)
        docs = [d for d in self._fallback.documents.values() if d.get("user_id") == user_id]
        docs.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        return docs

    async def update_document_status(
        self, doc_id: str, status: str, chunk_count: int = 0, indexed_at: Optional[str] = None
    ) -> None:
        """Update indexing status."""
        updates: Dict[str, Any] = {"status": status, "chunk_count": chunk_count}
        if indexed_at:
            updates["indexed_at"] = indexed_at
        if self.is_connected and self.db is not None:
            await self.db.documents.update_one({"id": doc_id}, {"$set": updates})
        elif doc_id in self._fallback.documents:
            self._fallback.documents[doc_id].update(updates)

    async def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Delete document and its chunks."""
        if self.is_connected and self.db is not None:
            res = await self.db.documents.delete_one({"id": doc_id, "user_id": user_id})
            await self.db.document_chunks.delete_many({"document_id": doc_id, "user_id": user_id})
            return res.deleted_count > 0
        if doc_id in self._fallback.documents and self._fallback.documents[doc_id].get("user_id") == user_id:
            del self._fallback.documents[doc_id]
            self._fallback.document_chunks = [
                c for c in self._fallback.document_chunks if c.get("document_id") != doc_id
            ]
            return True
        return False

    async def insert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunk records."""
        if not chunks:
            return
        if self.is_connected and self.db is not None:
            await self.db.document_chunks.insert_many([dict(c) for c in chunks])
        else:
            self._fallback.document_chunks.extend(chunks)

    # ── Conversation Operations ───────────────────────────────────────────────
    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation session."""
        conv_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        doc = {
            "id": conv_id,
            "user_id": user_id,
            "title": title or "New Conversation",
            "created_at": now,
            "updated_at": now,
            "is_active": True,
            "metadata": {},
        }
        if self.is_connected and self.db is not None:
            await self.db.conversations.insert_one(dict(doc))
        else:
            self._fallback.conversations[conv_id] = doc
        return doc

    async def list_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """List active conversations for a user."""
        if self.is_connected and self.db is not None:
            cursor = self.db.conversations.find(
                {"user_id": user_id, "is_active": True}, {"_id": 0}
            ).sort("updated_at", -1)
            return await cursor.to_list(length=100)
        convs = [
            c for c in self._fallback.conversations.values()
            if c.get("user_id") == user_id and c.get("is_active", True)
        ]
        convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return convs

    async def get_conversation(self, conv_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID."""
        if self.is_connected and self.db is not None:
            return await self.db.conversations.find_one({"id": conv_id, "user_id": user_id}, {"_id": 0})
        c = self._fallback.conversations.get(conv_id)
        if c and c.get("user_id") == user_id:
            return dict(c)
        return None

    async def add_message(
        self, conv_id: str, user_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Append a message to a conversation."""
        msg_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        msg = {
            "id": msg_id,
            "conversation_id": conv_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": now,
            "metadata": metadata or {},
        }
        if self.is_connected and self.db is not None:
            await self.db.conversation_messages.insert_one(dict(msg))
            await self.db.conversations.update_one(
                {"id": conv_id}, {"$set": {"updated_at": now}}
            )
        else:
            self._fallback.messages.append(msg)
            if conv_id in self._fallback.conversations:
                self._fallback.conversations[conv_id]["updated_at"] = now
        return msg

    async def get_messages(self, conv_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chronological messages for a conversation."""
        if self.is_connected and self.db is not None:
            cursor = self.db.conversation_messages.find(
                {"conversation_id": conv_id}, {"_id": 0}
            ).sort("created_at", 1)
            return await cursor.to_list(length=limit)
        msgs = [m for m in self._fallback.messages if m.get("conversation_id") == conv_id]
        msgs.sort(key=lambda x: x.get("created_at", ""))
        return msgs[-limit:]

    # ── Memory Operations ─────────────────────────────────────────────────────
    async def insert_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store memory record."""
        mem = dict(memory_data)
        mem.setdefault("id", str(uuid.uuid4()))
        mem.setdefault("created_at", datetime.datetime.utcnow().isoformat())
        mem.setdefault("updated_at", mem["created_at"])
        mem.setdefault("is_active", True)
        if self.is_connected and self.db is not None:
            await self.db.memory_records.insert_one(dict(mem))
        else:
            self._fallback.memory_records[mem["id"]] = mem
        return mem

    async def list_memories(
        self, user_id: str, memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List memory records for a user."""
        q: Dict[str, Any] = {"user_id": user_id, "is_active": True}
        if memory_type:
            q["memory_type"] = memory_type
        if self.is_connected and self.db is not None:
            cursor = self.db.memory_records.find(q, {"_id": 0}).sort("created_at", -1)
            return await cursor.to_list(length=200)
        mems = [
            m for m in self._fallback.memory_records.values()
            if m.get("user_id") == user_id and m.get("is_active", True)
            and (not memory_type or m.get("memory_type") == memory_type)
        ]
        mems.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return mems

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Soft-delete memory record."""
        if self.is_connected and self.db is not None:
            res = await self.db.memory_records.update_one(
                {"id": memory_id, "user_id": user_id}, {"$set": {"is_active": False}}
            )
            return res.modified_count > 0
        if memory_id in self._fallback.memory_records:
            m = self._fallback.memory_records[memory_id]
            if m.get("user_id") == user_id:
                m["is_active"] = False
                return True
        return False

    # ── Analytics Operations ──────────────────────────────────────────────────
    async def insert_analytics(self, analytics_data: Dict[str, Any]) -> None:
        """Store query analytics record."""
        rec = dict(analytics_data)
        rec.setdefault("id", str(uuid.uuid4()))
        rec.setdefault("created_at", datetime.datetime.utcnow().isoformat())
        if self.is_connected and self.db is not None:
            await self.db.query_analytics.insert_one(dict(rec))
        else:
            self._fallback.query_analytics.append(rec)

    async def get_analytics_summary(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate analytics metrics."""
        if self.is_connected and self.db is not None:
            match_q = {"user_id": user_id} if user_id else {}
            pipeline = [
                {"$match": match_q},
                {
                    "$group": {
                        "_id": None,
                        "total_queries": {"$sum": 1},
                        "avg_execution_time_ms": {"$avg": "$execution_time_ms"},
                        "total_cost_usd": {"$sum": "$estimated_cost_usd"},
                        "cache_hits": {"$sum": {"$cond": ["$cache_hit", 1, 0]}},
                    }
                },
            ]
            agg = await self.db.query_analytics.aggregate(pipeline).to_list(length=1)
            if agg:
                item = agg[0]
                total = item.get("total_queries", 0)
                hits = item.get("cache_hits", 0)
                return {
                    "total_queries": total,
                    "avg_latency_ms": round(item.get("avg_execution_time_ms") or 0.0, 2),
                    "total_estimated_cost_usd": round(item.get("total_cost_usd") or 0.0, 6),
                    "cache_hit_rate": round(hits / total, 3) if total > 0 else 0.0,
                }
            return {"total_queries": 0, "avg_latency_ms": 0.0, "total_estimated_cost_usd": 0.0, "cache_hit_rate": 0.0}

        # Fallback calculation
        recs = [
            r for r in self._fallback.query_analytics
            if user_id is None or r.get("user_id") == user_id
        ]
        total = len(recs)
        if not total:
            return {"total_queries": 0, "avg_latency_ms": 0.0, "total_estimated_cost_usd": 0.0, "cache_hit_rate": 0.0}
        avg_time = sum(r.get("execution_time_ms", 0.0) or 0.0 for r in recs) / total
        total_cost = sum(r.get("estimated_cost_usd", 0.0) or 0.0 for r in recs)
        cache_hits = sum(1 for r in recs if r.get("cache_hit"))
        return {
            "total_queries": total,
            "avg_latency_ms": round(avg_time, 2),
            "total_estimated_cost_usd": round(total_cost, 6),
            "cache_hit_rate": round(cache_hits / total, 3),
        }


# Global MongoDB manager instance
mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_db_name)
