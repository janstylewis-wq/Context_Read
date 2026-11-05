"""Database schema (Alembic owns migrations; the web app reads these tables through Drizzle).

Owners: documents, folders and chats belong to a user or a teamspace (owner_kind + owner_id).
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .ids import new_id


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


def _id() -> Mapped[str]:
    return mapped_column(Text, primary_key=True, default=new_id)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


def _updated() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


OWNER_KINDS = "owner_kind IN ('user', 'teamspace')"


# ---------- identity (Auth.js-compatible) ----------


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = _id()
    name: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text, unique=True)
    email_verified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    image: Mapped[str | None] = mapped_column(Text)
    password_hash: Mapped[str | None] = mapped_column(Text)
    user_type: Mapped[str] = mapped_column(Text, server_default="normal")
    is_admin: Mapped[bool] = mapped_column(Boolean, server_default="false")
    allow_data_collection: Mapped[bool] = mapped_column(Boolean, server_default="true")
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (UniqueConstraint("provider", "provider_account_id"),)
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(Text)
    provider_account_id: Mapped[str] = mapped_column(Text)
    refresh_token: Mapped[str | None] = mapped_column(Text)
    access_token: Mapped[str | None] = mapped_column(Text)
    expires_at: Mapped[int | None] = mapped_column(Integer)
    token_type: Mapped[str | None] = mapped_column(Text)
    scope: Mapped[str | None] = mapped_column(Text)
    id_token: Mapped[str | None] = mapped_column(Text)
    session_state: Mapped[str | None] = mapped_column(Text)


class AuthSession(Base):
    __tablename__ = "sessions"
    session_token: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AuthCode(Base):
    """A one-time sign-in code and magic link (purpose 'login') or password-reset link ('reset').
    Only hashes are stored; a code dies after MAX_ATTEMPTS wrong guesses, expiry or first use."""

    __tablename__ = "auth_codes"
    __table_args__ = (
        CheckConstraint("purpose IN ('login', 'reset')"),
        Index("ix_auth_codes_email_created", "email", "created_at"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(Text)
    purpose: Mapped[str] = mapped_column(Text)
    code_hash: Mapped[str | None] = mapped_column(Text)
    link_hash: Mapped[str] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()


class VerificationToken(Base):
    __tablename__ = "verification_tokens"
    __table_args__ = (PrimaryKeyConstraint("identifier", "token"),)
    identifier: Mapped[str] = mapped_column(Text)
    token: Mapped[str] = mapped_column(Text)
    expires: Mapped[datetime] = mapped_column(DateTime(timezone=True))


# ---------- teamspaces ----------


class Teamspace(Base):
    __tablename__ = "teamspaces"
    id: Mapped[str] = _id()
    name: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = _created()


class TeamspaceMember(Base):
    __tablename__ = "teamspace_members"
    __table_args__ = (
        PrimaryKeyConstraint("teamspace_id", "user_id"),
        CheckConstraint("role IN ('owner', 'admin', 'member')"),
    )
    teamspace_id: Mapped[str] = mapped_column(ForeignKey("teamspaces.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(Text, server_default="member")
    created_at: Mapped[datetime] = _created()


class TeamspaceInvitation(Base):
    __tablename__ = "teamspace_invitations"
    id: Mapped[str] = _id()
    teamspace_id: Mapped[str] = mapped_column(ForeignKey("teamspaces.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text, server_default="member")
    token: Mapped[str] = mapped_column(Text, unique=True)
    invited_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()


# ---------- keys, plans, usage ----------


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (CheckConstraint("product IN ('cloud', 'search')"),)
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str | None] = mapped_column(Text)
    product: Mapped[str] = mapped_column(Text)
    prefix: Mapped[str] = mapped_column(Text)
    secret_sha256: Mapped[str] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = _created()
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ---------- OAuth for MCP clients ----------


class OAuthClient(Base):
    """A dynamically registered MCP client (RFC 7591); `info` is its registration as issued."""

    __tablename__ = "oauth_clients"
    id: Mapped[str] = mapped_column(Text, primary_key=True)  # the client_id
    info: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = _created()


class OAuthRequest(Base):
    """An /authorize request waiting for the user's consent in the web app."""

    __tablename__ = "oauth_requests"
    id: Mapped[str] = _id()
    client_id: Mapped[str] = mapped_column(ForeignKey("oauth_clients.id", ondelete="CASCADE"))
    params: Mapped[dict[str, Any]]
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()


class OAuthCode(Base):
    __tablename__ = "oauth_codes"
    code_sha256: Mapped[str] = mapped_column(Text, primary_key=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("oauth_clients.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    grant_id: Mapped[str] = mapped_column(Text)
    params: Mapped[dict[str, Any]]
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OAuthToken(Base):
    """Access and refresh tokens, stored as hashes. A grant (one consent) groups the tokens that
    refreshing rotates through, so revoking one revokes them all."""

    __tablename__ = "oauth_tokens"
    __table_args__ = (
        CheckConstraint("kind IN ('access', 'refresh')"),
        Index("ix_oauth_tokens_user", "user_id", "client_id"),
        Index("ix_oauth_tokens_grant", "grant_id"),
    )
    token_sha256: Mapped[str] = mapped_column(Text, primary_key=True)
    kind: Mapped[str] = mapped_column(Text)
    client_id: Mapped[str] = mapped_column(ForeignKey("oauth_clients.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    grant_id: Mapped[str] = mapped_column(Text)
    scopes: Mapped[list[Any]]
    resource: Mapped[str | None] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()


class Subscription(Base):
    __tablename__ = "subscriptions"
    __table_args__ = (CheckConstraint(OWNER_KINDS),)
    id: Mapped[str] = _id()
    owner_kind: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[str] = mapped_column(Text, index=True)
    plan: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    billing_period: Mapped[str] = mapped_column(Text, server_default="monthly")
    provider_customer_id: Mapped[str | None] = mapped_column(Text)
    provider_subscription_id: Mapped[str | None] = mapped_column(Text, unique=True)
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()


class CreditLedger(Base):
    __tablename__ = "credit_ledger"
    __table_args__ = (CheckConstraint(OWNER_KINDS),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    owner_kind: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[str] = mapped_column(Text, index=True)
    delta_usd: Mapped[Decimal] = mapped_column(Numeric(14, 6))
    reason: Mapped[str] = mapped_column(Text)
    ref: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class UsageEvent(Base):
    __tablename__ = "usage_events"
    __table_args__ = (
        CheckConstraint(OWNER_KINDS),
        CheckConstraint("kind IN ('page', 'chat', 'search', 'storage')"),
        Index("ix_usage_events_owner_kind_created", "owner_id", "kind", "created_at"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    owner_kind: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    qty: Mapped[int] = mapped_column(Integer)
    file_id: Mapped[str | None] = mapped_column(Text)
    ref: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


# ---------- documents ----------


class Folder(Base):
    __tablename__ = "folders"
    __table_args__ = (CheckConstraint(OWNER_KINDS), Index("ix_folders_owner", "owner_kind", "owner_id"))
    id: Mapped[str] = _id()
    owner_kind: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[str] = mapped_column(Text)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("folders.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class MetadataSchema(Base):
    """Declared document-metadata fields for an owner's whole library (scope 'root') or for one
    top-level folder (scope = that folder's id); subfolders inherit their top-level folder's."""

    __tablename__ = "metadata_schemas"
    __table_args__ = (CheckConstraint(OWNER_KINDS), PrimaryKeyConstraint("owner_kind", "owner_id", "scope"))
    owner_kind: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(Text)
    fields: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = _created()


class File(Base):
    __tablename__ = "files"
    __table_args__ = (
        CheckConstraint(OWNER_KINDS),
        CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')"),
        Index("ix_files_owner", "owner_kind", "owner_id", "created_at"),
        Index("ix_files_owner_hash", "owner_kind", "owner_id", "doc_hash"),
        UniqueConstraint("owner_kind", "owner_id", "file_name", name="uq_files_owner_name"),
    )
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: new_id("doc-"))
    owner_kind: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[str] = mapped_column(Text)
    created_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    folder_id: Mapped[str | None] = mapped_column(ForeignKey("folders.id", ondelete="SET NULL"), index=True)
    file_name: Mapped[str] = mapped_column(Text)
    storage_key: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    config: Mapped[dict[str, Any]] = mapped_column(server_default="{}")
    page_num: Mapped[int | None] = mapped_column(Integer)
    doc_hash: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    structure: Mapped[list[Any] | None]
    md_tree: Mapped[list[Any] | None]
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata")
    has_block_ref: Mapped[bool] = mapped_column(Boolean, server_default="false")
    thumbnail_key: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    index_version: Mapped[int] = mapped_column(Integer, server_default="1")
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (
        PrimaryKeyConstraint("file_id", "page"),
        CheckConstraint("read_mode IN ('text', 'image', 'vlm')", name="pages_read_mode_check"),
        Index("ix_pages_text_tsv", "text_tsv", postgresql_using="gin"),
    )
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    page: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    image_key: Mapped[str | None] = mapped_column(Text)
    read_mode: Mapped[str] = mapped_column(Text, server_default="text")
    text: Mapped[str | None] = mapped_column(Text)
    text_tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', coalesce(text, ''))", persisted=True)
    )


class Block(Base):
    __tablename__ = "blocks"
    __table_args__ = (
        PrimaryKeyConstraint("file_id", "block_id"),
        Index("ix_blocks_file_page", "file_id", "page", "ord"),
    )
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    block_id: Mapped[str] = mapped_column(Text)
    page: Mapped[int] = mapped_column(Integer)
    ord: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(Text)
    text: Mapped[str | None] = mapped_column(Text)
    bbox: Mapped[list[int]] = mapped_column(ARRAY(Integer))


# ---------- background jobs ----------


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("status IN ('queued', 'running', 'done', 'failed')"),
        Index("ix_jobs_claim", "status", "run_after"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(server_default="{}")
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, server_default="3")
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    locked_by: Mapped[str | None] = mapped_column(Text)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Retrieval(Base):
    """A tree-search retrieval request (REST /retrieval/), answered asynchronously by the worker."""

    __tablename__ = "retrievals"
    __table_args__ = (CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')"),)
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: new_id("ret_"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    query: Mapped[str] = mapped_column(Text)
    thinking: Mapped[bool] = mapped_column(Boolean, server_default="false")
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    result: Mapped[list[Any] | None]
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()


class LlmSpend(Base):
    """Ledger behind the hard LLM budget cap (see llm/budget.py)."""

    __tablename__ = "llm_spend"
    __table_args__ = (CheckConstraint("status IN ('reserved', 'done', 'failed')"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model: Mapped[str] = mapped_column(Text)
    purpose: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    cost_source: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(Text, server_default="openai")


# ---------- chat & sharing ----------


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_user_updated", "user_id", "updated_at"),)
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    teamspace_id: Mapped[str | None] = mapped_column(ForeignKey("teamspaces.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata")
    share_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')"),
        Index("ix_chat_messages_session_created", "session_id", "created_at"),
    )
    id: Mapped[str] = _id()
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(Text)
    parts: Mapped[list[Any]]
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata")
    created_at: Mapped[datetime] = _created()


class Share(Base):
    __tablename__ = "shares"
    __table_args__ = (
        CheckConstraint("target_type IN ('chat', 'doc', 'folder')"),
        Index("ix_shares_target", "target_type", "target_id"),
    )
    id: Mapped[str] = _id()
    target_type: Mapped[str] = mapped_column(Text)
    target_id: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = _created()
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ShareDocument(Base):
    """Documents a share exposes; a chat share covers only the documents its answers cite."""

    __tablename__ = "share_documents"
    __table_args__ = (PrimaryKeyConstraint("share_id", "file_id"),)
    share_id: Mapped[str] = mapped_column(ForeignKey("shares.id", ondelete="CASCADE"))
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))


class SavedShare(Base):
    __tablename__ = "saved_shares"
    __table_args__ = (UniqueConstraint("user_id", "share_id"),)
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    share_id: Mapped[str] = mapped_column(ForeignKey("shares.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(Text, server_default="user_share")
    status: Mapped[str] = mapped_column(Text, server_default="active")
    snapshot: Mapped[dict[str, Any] | None]
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ExploreItem(Base):
    __tablename__ = "explore_items"
    __table_args__ = (CheckConstraint("status IN ('draft', 'published', 'unpublished', 'taken_down')"),)
    id: Mapped[str] = _id()
    slug: Mapped[str] = mapped_column(Text, unique=True)
    share_id: Mapped[str] = mapped_column(ForeignKey("shares.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(Text)
    subtitle: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    cover_key: Mapped[str | None] = mapped_column(Text)
    icon_key: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[Any]] = mapped_column(server_default="[]")
    category: Mapped[str | None] = mapped_column(Text)
    featured: Mapped[bool] = mapped_column(Boolean, server_default="false")
    sort_order: Mapped[int] = mapped_column(Integer, server_default="0")
    status: Mapped[str] = mapped_column(Text, server_default="draft")
    recommended_prompts: Mapped[list[Any]] = mapped_column(server_default="[]")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()


# ---------- Page Search API ----------


class SearchCollection(Base):
    __tablename__ = "search_collections"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: new_id("col_"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class SearchCollectionDocument(Base):
    __tablename__ = "search_collection_documents"
    __table_args__ = (PrimaryKeyConstraint("collection_id", "file_id"),)
    collection_id: Mapped[str] = mapped_column(ForeignKey("search_collections.id", ondelete="CASCADE"))
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), index=True)


class SearchRequest(Base):
    __tablename__ = "search_requests"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: new_id("srch_"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    questions: Mapped[list[Any]]
    params: Mapped[dict[str, Any]] = mapped_column(server_default="{}")
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), server_default="0")
    result: Mapped[dict[str, Any] | None]
    created_at: Mapped[datetime] = _created()
