"""Folders (workspaces) and their document-metadata schemas."""

from typing import Any

from sqlalchemy import Engine, func, select

from .db import session_scope
from .models import File, Folder, MetadataSchema

FIELD_TYPES = {"string": str, "number": (int, float), "boolean": bool}
MAX_FIELDS = 16


class FolderError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def _folder_out(s, f: Folder) -> dict:
    return {
        "id": f.id,
        "name": f.name,
        "description": f.description,
        "parent_folder_id": f.parent_id,
        "created_at": f.created_at.isoformat(),
        "file_count": s.scalar(select(func.count()).select_from(File).where(File.folder_id == f.id)),
        "children_count": s.scalar(select(func.count()).select_from(Folder).where(Folder.parent_id == f.id)),
    }


def _owned(s, owner: tuple[str, str], folder_id: str) -> Folder:
    f = s.get(Folder, folder_id)
    if f is None or (f.owner_kind, f.owner_id) != owner:
        raise FolderError(404, "folder_not_found", "Folder not found.")
    return f


def create_folder(
    owner: tuple[str, str],
    name: str,
    description: str | None = None,
    parent_folder_id: str | None = None,
    engine: Engine | None = None,
) -> dict:
    name = (name or "").strip()
    if not name or any(c in name for c in '/\\:*?"<>|'):
        raise FolderError(400, "invalid_name", 'Folder name cannot be empty or contain / \\ : * ? " < > |')
    with session_scope(engine) as s:
        if parent_folder_id is not None:
            _owned(s, owner, parent_folder_id)
        f = Folder(
            owner_kind=owner[0],
            owner_id=owner[1],
            name=name,
            description=description,
            parent_id=parent_folder_id,
        )
        s.add(f)
        s.flush()
        s.refresh(f)
        return {"folder": _folder_out(s, f)}


def list_folders(
    owner: tuple[str, str], parent_folder_id: str | None = None, engine: Engine | None = None
) -> dict:
    with session_scope(engine) as s:
        q = select(Folder).where(Folder.owner_kind == owner[0], Folder.owner_id == owner[1])
        if parent_folder_id == "root":
            q = q.where(Folder.parent_id.is_(None))
        elif parent_folder_id is not None:
            q = q.where(Folder.parent_id == parent_folder_id)
        folders = [_folder_out(s, f) for f in s.scalars(q.order_by(Folder.created_at))]
        return {"folders": folders, "total": len(folders)}


def top_level(s, folder: Folder) -> Folder:
    while folder.parent_id is not None:
        folder = s.get(Folder, folder.parent_id)
    return folder


def _validate_fields(fields: Any) -> dict:
    if not isinstance(fields, dict) or len(fields) > MAX_FIELDS:
        raise FolderError(400, "invalid_schema", f"fields must be an object of at most {MAX_FIELDS} fields")
    for key, spec in fields.items():
        if not isinstance(spec, dict) or spec.get("type") not in FIELD_TYPES:
            raise FolderError(
                400, "invalid_schema", f"field {key!r} needs a type of string, number or boolean"
            )
    return fields


def create_schema(owner: tuple[str, str], folder_id: str, fields: dict, engine: Engine | None = None) -> dict:
    fields = _validate_fields(fields)
    with session_scope(engine) as s:
        if folder_id != "root":
            folder = _owned(s, owner, folder_id)
            if folder.parent_id is not None:
                raise FolderError(400, "not_top_level", "Metadata schemas are declared on top-level folders.")
        if s.get(MetadataSchema, (owner[0], owner[1], folder_id)) is not None:
            raise FolderError(
                409, "schema_exists", "A metadata schema already exists; delete it first to change it."
            )
        s.add(MetadataSchema(owner_kind=owner[0], owner_id=owner[1], scope=folder_id, fields=fields))
    return {"folder_id": folder_id, "schema": {"fields": fields}}


def applicable_schema(s, owner: tuple[str, str], folder_id: str | None) -> tuple[str, MetadataSchema | None]:
    """(workspace_id, schema) for a folder: its top-level folder's schema, else the root schema."""
    if folder_id not in (None, "root"):
        workspace = top_level(s, _owned(s, owner, folder_id))
        schema = s.get(MetadataSchema, (owner[0], owner[1], workspace.id))
        if schema is not None:
            return workspace.id, schema
    return "root", s.get(MetadataSchema, (owner[0], owner[1], "root"))


def get_schema(owner: tuple[str, str], folder_id: str, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        workspace_id, schema = applicable_schema(s, owner, folder_id)
        if schema is None:
            raise FolderError(404, "schema_not_found", "No metadata schema applies to this folder.")
        return {
            "folder_id": folder_id,
            "workspace_id": workspace_id,
            "inherited": workspace_id != folder_id,
            "schema": {"fields": schema.fields},
        }


def delete_schema(owner: tuple[str, str], folder_id: str, engine: Engine | None = None) -> dict:
    with session_scope(engine) as s:
        schema = s.get(MetadataSchema, (owner[0], owner[1], folder_id))
        if schema is None:
            raise FolderError(404, "schema_not_found", "This folder declares no metadata schema.")
        s.delete(schema)
    return {"folder_id": folder_id, "deleted": True}


def validate_metadata(
    owner: tuple[str, str], folder_id: str | None, metadata: dict | None, engine: Engine | None = None
) -> None:
    if metadata is None:
        return
    if not isinstance(metadata, dict) or len(metadata) > MAX_FIELDS:
        raise FolderError(400, "invalid_metadata", f"metadata must be an object of at most {MAX_FIELDS} keys")
    with session_scope(engine) as s:
        _, schema = applicable_schema(s, owner, folder_id)
        fields = schema.fields if schema else {}
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise FolderError(
                400, "invalid_metadata", f"metadata {key!r} must be a string, number or boolean"
            )
        spec = fields.get(key)
        if spec:
            want = FIELD_TYPES[spec["type"]]
            if value is not None and (
                not isinstance(value, want) or (spec["type"] == "number" and isinstance(value, bool))
            ):
                raise FolderError(400, "invalid_metadata", f"metadata {key!r} must be a {spec['type']}")


def folder_tree(owner: tuple[str, str], engine: Engine | None = None) -> list[dict]:
    """Nested folders with document counts, as the web sidebar shows them."""
    with session_scope(engine) as s:
        rows = list(
            s.scalars(
                select(Folder)
                .where(Folder.owner_kind == owner[0], Folder.owner_id == owner[1])
                .order_by(Folder.name)
            )
        )
        counts = dict(
            s.execute(
                select(File.folder_id, func.count())
                .where(File.owner_kind == owner[0], File.owner_id == owner[1], File.folder_id.is_not(None))
                .group_by(File.folder_id)
            ).all()
        )
    nodes = {
        f.id: {
            "id": f.id,
            "name": f.name,
            "parentId": f.parent_id,
            "fileCount": counts.get(f.id, 0),
            "children": [],
        }
        for f in rows
    }
    roots = []
    for n in nodes.values():
        (nodes[n["parentId"]]["children"] if n["parentId"] in nodes else roots).append(n)
    return roots


def update_folder(
    owner: tuple[str, str],
    folder_id: str,
    name: str | None = None,
    parent_id=None,
    move: bool = False,
    engine: Engine | None = None,
) -> dict:
    with session_scope(engine) as s:
        f = _owned(s, owner, folder_id)
        if name is not None:
            name = name.strip()
            if not name or any(c in name for c in '/\\:*?"<>|'):
                raise FolderError(
                    400, "invalid_name", 'Folder name cannot be empty or contain / \\ : * ? " < > |'
                )
            f.name = name
        if move:
            cursor = parent_id
            while cursor is not None:  # refuse moving a folder into itself or its own subtree
                if cursor == folder_id:
                    raise FolderError(400, "invalid_move", "A folder cannot move into itself.")
                cursor = _owned(s, owner, cursor).parent_id
            f.parent_id = parent_id
        s.flush()
        s.refresh(f)
        return _folder_out(s, f)


def delete_folder(owner: tuple[str, str], folder_id: str, engine: Engine | None = None) -> dict:
    """Deletes the folder and its subfolders; their documents move to the top level, not deleted."""
    with session_scope(engine) as s:
        _owned(s, owner, folder_id)
        ids, frontier = [folder_id], [folder_id]
        while frontier:
            frontier = list(s.scalars(select(Folder.id).where(Folder.parent_id.in_(frontier))))
            ids += frontier
        moved = 0
        for f in s.scalars(select(File).where(File.folder_id.in_(ids))):
            f.folder_id, moved = None, moved + 1
        s.flush()
        s.delete(s.get(Folder, folder_id))
    return {"deleted": len(ids), "documentsMoved": moved}
